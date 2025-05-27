import asyncio
import copy
import logging
import os
import signal
import socket
import uuid
from typing import AsyncGenerator, Optional

from components.worker import VllmPrefillWorker
from utils.args import parse_vllm_args
from utils.protocol import MyRequestOutput, PreprocessedRequest, vLLMGenerateRequest
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.inputs import TokensPrompt
from vllm.sampling_params import SamplingParams

from dynamo.llm import ModelType, register_llm
from dynamo.sdk import async_on_start, dynamo_context, endpoint, service

logger = logging.getLogger(__name__)


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class DecodeWorkerAndPrefillLoadBalancer:
    """
    Follows the example of SimpleLoadBalancer to load balance to prefill, but works
    in a multi-node setup where decode and prefill are not collocated.

    The generate() flow is:
        1. Select a (remote) prefill worker and send the request to it.
        2. Once the prefill response is received, run decode locally and stream back
           the response.
    """

    def __init__(self) -> None:
        class_name = self.__class__.__name__
        self.engine_args = parse_vllm_args(class_name, "")

        signal.signal(signal.SIGTERM, self.shutdown_vllm_engine)
        signal.signal(signal.SIGINT, self.shutdown_vllm_engine)

        self.set_side_channel_port()

        model_config = self.engine_args.create_model_config()
        self.default_sampling_params = model_config.get_diff_sampling_param()

        # Lazily initialized
        self.prefill_client = None

    def shutdown_vllm_engine(self, signum, frame):
        """Shutdown the background loop"""
        logger.info(f"Received signal {signum}, shutting down")
        loop = asyncio.get_event_loop()
        try:
            self.engine_client.close()
            logger.info("VllmWorker shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            loop.stop()

    def set_side_channel_port(self, port: Optional[int] = None):
        """vLLM V1 NixlConnector creates a side channel to exchange metadata with other NIXL connectors.
        This sets the port number for the side channel.
        """
        if port is None:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))  # Bind to a free port provided by the host.
                port = s.getsockname()[1]  # Get the port number assigned.
        logger.debug("Setting VLLM_NIXL_SIDE_CHANNEL_PORT to %s", port)
        os.environ["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(port)

    @async_on_start
    async def async_init(self) -> None:
        # Start local decode instance
        self._engine_context = build_async_engine_client_from_engine_args(
            self.engine_args
        )
        if self._engine_context is not None:
            self.engine_client = await self._engine_context.__aenter__()
        else:
            raise RuntimeError("Failed to initialize engine client")

        logger.info("Decode instance has been initialized")

        runtime = dynamo_context["runtime"]
        comp_ns, comp_name = DecodeWorkerAndPrefillLoadBalancer.dynamo_address()
        endpoint_name = "generate"

        for served_model_name in self.engine_args.served_model_name:
            logger.info(
                f"Registering endpoint {endpoint_name} with model {self.engine_args.model} and served_model_name {served_model_name}"
            )
            endpoint = (
                runtime.namespace(comp_ns).component(comp_name).endpoint(endpoint_name)
            )
            await register_llm(
                ModelType.Backend, endpoint, self.engine_args.model, served_model_name
            )

        logger.info("Prefill load balancer has been initialized")

    async def send_request_to_prefill(
        self, request: vLLMGenerateRequest
    ) -> MyRequestOutput:
        prefill_request = copy.deepcopy(request)
        extra_args = prefill_request.sampling_params.extra_args or {}
        extra_args["kv_transfer_params"] = {
            "do_remote_decode": True,
        }
        prefill_request.sampling_params.extra_args = extra_args
        prefill_request.sampling_params.max_tokens = 1
        prefill_request.sampling_params.min_tokens = 1

        if not self.prefill_client:
            runtime = dynamo_context["runtime"]
            comp_ns, comp_name = VllmPrefillWorker.dynamo_address()

            self.prefill_client = (
                await runtime.namespace(comp_ns)
                .component(comp_name)
                .endpoint("generate")
                .client()
            )

        # TODO Don't round robin
        async for prefill_response in await self.prefill_client.round_robin(
            prefill_request.model_dump_json()
        ):
            logger.info(f"Prefill response: {prefill_response}")
            return MyRequestOutput.model_validate_json(prefill_response.data())

    async def run_decode(
        self,
        request: vLLMGenerateRequest,
        prefill_response: Optional[MyRequestOutput] = None,
    ) -> AsyncGenerator[MyRequestOutput, None]:
        logger.debug("Sending request to decode")

        decode_request = copy.deepcopy(request)

        if prefill_response:
            extra_args = decode_request.sampling_params.extra_args or {}
            extra_args["kv_transfer_params"] = prefill_response.kv_transfer_params
            decode_request.sampling_params.extra_args = extra_args

        logger.debug("Decode request: %s", decode_request.model_dump_json())

        gen = self.engine_client.generate(
            prompt=decode_request.prompt,
            sampling_params=decode_request.sampling_params,
            request_id=decode_request.request_id,
        )

        async for response in gen:
            yield MyRequestOutput(
                request_id=response.request_id,
                prompt=response.prompt,
                prompt_token_ids=response.prompt_token_ids,
                prompt_logprobs=response.prompt_logprobs,
                outputs=response.outputs,
                finished=response.finished,
                metrics=response.metrics,
                kv_transfer_params=response.kv_transfer_params,
            )

    async def _stream_response(self, gen: AsyncGenerator[MyRequestOutput, None]):
        num_output_tokens_so_far = 0
        async for res in gen:
            logger.debug("Decode response: %s", res.model_dump_json())
            # res is our MyRequestOutput

            # This is the expected way for a request to end.
            # The new token ID will be eos, don't forward it.
            if res.finished:
                yield {"finish_reason": "stop", "token_ids": []}
                break

            if not res.outputs:
                yield {"finish_reason": "error", "token_ids": []}
                break

            output = res.outputs[0]
            next_total_toks = len(output.token_ids)
            out = {"token_ids": output.token_ids[num_output_tokens_so_far:]}
            if output.finish_reason:
                out["finish_reason"] = output.finish_reason
            if output.stop_reason:
                out["stop_reason"] = output.stop_reason
            yield out
            num_output_tokens_so_far = next_total_toks

    @endpoint()
    async def generate(self, request: PreprocessedRequest):
        vllm_request = self._create_vllm_request(request)
        logger.info(f"Sending request to prefill: {vllm_request.model_dump_json()}")

        prefill_response = await self.send_request_to_prefill(vllm_request)

        async for res in self._stream_response(
            self.run_decode(vllm_request, prefill_response)
        ):
            yield res

    def _create_vllm_request(self, request: PreprocessedRequest) -> vLLMGenerateRequest:
        request_id = str(uuid.uuid4().hex)

        prompt = TokensPrompt(prompt_token_ids=request.token_ids)

        sampling_params = SamplingParams(**self.default_sampling_params)
        for key, value in request.sampling_options.model_dump().items():
            if not value:
                continue
            if hasattr(sampling_params, key):
                setattr(sampling_params, key, value)

        max_tokens = request.stop_conditions.max_tokens
        if max_tokens:
            sampling_params.max_tokens = max_tokens

        return vLLMGenerateRequest(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
        )
