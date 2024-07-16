import json
import os
from abc import ABC, abstractmethod
from typing import List
from warnings import warn

import openai
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
 
# anthropic
try:
    import anthropic

    from evalplus.gen.util import anthropic_request
except ImportError:
    warn("Anthropic decoder will not work. Fix by `pip install anthropic`")

# mistral.ai
try:
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage
except ImportError:
    warn("MistralAI decoder will not work. Fix by `pip install mistralai`")

# vllm
try:
    from vllm import LLM, SamplingParams
except ImportError:
    warn("VLLM decoder will not work. Fix by `pip install vllm`")

# hf-based
try:
    from stop_sequencer import StopSequencer
except ImportError:
    warn("HF-based decoder will not work. Fix by `pip install vllm`")
    
# google
try:
    import google.generativeai as genai
except ImportError:
    warn("google.generativeai will not work. Fix by `pip install google`")

from race.codegen.utils import get_openai_response, get_genai_response
from race.codegen.prompts import construct_prompt, construct_prompt_simple


EOS = [
    "<|endoftext|>",
    "<|endofmask|>",
    "</s>",
    "\nif __name__",
    "\ndef main(",
    "\nprint(",
]


def extra_eos_for_direct_completion(dataset) -> List[str]:
    if dataset.lower() == "humaneval":
        return ["\ndef ", "\nclass ", "\nimport ", "\nfrom ", "\nassert "]
    elif dataset.lower() == "mbpp":
        # return ['\n"""', "\nassert"]
        return ['\n"""', "\nassert", "\n\tassert"]  # TODO
    elif dataset.lower() == "classeval":
        return []
    elif dataset.lower() == "leetcode":
        return []
    elif dataset.lower() == "leetcode_efficiency":
        return []
    raise ValueError(f"Unknown dataset: {dataset}")


class DecoderBase(ABC):
    def __init__(
        self,
        name: str,
        batch_size: int = 1,
        temperature: float = 0.8,
        max_new_tokens: int = 2048,
        dtype: str = "bfloat16",  # default
        trust_remote_code: bool = True,
    ) -> None:
        print("Initializing a decoder model: {} ...".format(name))
        self.name = name
        self.batch_size = batch_size
        self.temperature = temperature
        self.eos = EOS
        self.skip_special_tokens = False
        self.max_new_tokens = max_new_tokens
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code

    @abstractmethod
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        pass

    @abstractmethod
    def is_direct_completion(self) -> bool:
        pass

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


class VllmDecoder(DecoderBase):
    def __init__(self, name: str, dataset: str, tp: int, **kwargs) -> None:
        super().__init__(name, **kwargs)

        kwargs = {
            "tensor_parallel_size": torch.cuda.device_count(),  # int(os.getenv("VLLM_N_GPUS", tp))
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
        }
        print(kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        if self.tokenizer.chat_template is None:
            self.eos += extra_eos_for_direct_completion(dataset)
        print(f'self.eos: {self.eos}')  # TODO
        self.llm = LLM(model=name, max_model_len=2048, **kwargs)

    def is_direct_completion(self) -> bool:
        return self.tokenizer.chat_template is None

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be greater than 0!"
        batch_size = min(self.batch_size, num_samples)

        vllm_outputs = self.llm.generate(
            [prompt] * batch_size,
            SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                top_p=0.95 if do_sample else 1.0,
                stop=self.eos,
            ),
            use_tqdm=False,
        )

        gen_strs = [x.outputs[0].text.replace("\t", "    ") for x in vllm_outputs]
        
        if len(gen_strs) == 1:
            return gen_strs[0]
        else:
            return gen_strs


class GeneralVllmDecoder(VllmDecoder):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.eos += ["\n```\n"]
        print(f"EOS strings: {self.eos}")

    def codegen(
        self, 
        prompt: str, 
        dim: str, 
        do_sample: bool = True, 
        num_samples: int = 1,
        strategy: str = 'customized',
        **kwargs
    ) -> List[str]:
        prompt = construct_prompt(prompt, strategy, dim, self.tokenizer, **kwargs)
        
        # TODO
        print('-' * 50)
        print(prompt)
        print('-' * 50)
        
        result = VllmDecoder.codegen(self, prompt, do_sample, num_samples)
        print(result)
        
        return result
    
    
class GenaiDecoder(DecoderBase):
    def __init__(self, name: str, api_key=None, **kwargs) -> None:
        super().__init__(name, **kwargs)
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.name,
                                           system_instruction="You are a helpful assistant good at coding.")

    def codegen(
        self, 
        prompt: str, 
        dim: str, 
        do_sample: bool = True, 
        num_samples: int = 1,
        strategy: str = 'customized',
        **kwargs
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"
        batch_size = min(self.batch_size, num_samples)

        # construct prompt
        user_prompt = construct_prompt_simple(prompt, strategy, dim, **kwargs)

        # TODO
        print('-' * 50)
        print(user_prompt)
        print('-' * 50)
        
        outputs = []
        for i in range(batch_size):
            response = get_genai_response(self.model, user_prompt, self.temperature)
            print(response)
            outputs.append(response)

        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

    def is_direct_completion(self) -> bool:
        return False


class OpenAIChatDecoder(DecoderBase):
    def __init__(self, name: str, base_url=None, api_key=None, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.client = openai.OpenAI(base_url=base_url,
                                    api_key=api_key)

    def codegen(
        self, 
        prompt: str, 
        dim: str, 
        do_sample: bool = True, 
        num_samples: int = 1,
        strategy: str = 'customized',
        **kwargs
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"
        batch_size = min(self.batch_size, num_samples)

        # construct prompt
        system_prompt = "You are a helpful assistant good at coding."
        user_prompt = construct_prompt_simple(prompt, strategy, dim, **kwargs)

        # TODO
        print('-' * 50)
        print(user_prompt)
        print('-' * 50)

        ret = get_openai_response(
            self.client,
            model=self.name,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=self.temperature,
            n=batch_size
        )

        outputs = []
        for item in ret.choices:
            content = item.message.content
            outputs.append(content)

        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

    def is_direct_completion(self) -> bool:
        return False


class HfTorchDecoder(DecoderBase):
    def __init__(self, name: str, dataset: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        kwargs = {}
        kwargs["device_map"] = "auto"
        kwargs["trust_remote_code"] = self.trust_remote_code
        # string to torch dtype
        kwargs["torch_dtype"] = getattr(torch, self.dtype)
        self.skip_special_tokens = True

        print(f"{kwargs = }")

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        if self.tokenizer.chat_template is None:
            self.eos += extra_eos_for_direct_completion(dataset)
        print(f'self.eos: {self.eos}')  # TODO

        self.model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        self.model = self.model.to(self.device)

    def is_direct_completion(self) -> bool:
        return self.tokenizer.chat_template is not None

    @torch.inference_mode()
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if self.temperature == 0:
            assert not do_sample
            assert num_samples == 1

        input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.device
        )
        kwargs = {}
        if do_sample:
            kwargs["top_p"] = 0.95
            kwargs["temperature"] = self.temperature
            
        # TODO
        if not do_sample:
            kwargs["top_p"] = 1
            kwargs["temperature"] = 0

        stop_sequencer = StopSequencer(
            self.model,
            model_type="causal",  # or seq2seq
            tokenizer=self.tokenizer,
        )

        model = stop_sequencer.register_stop_texts(
            stop_texts=self.eos,
            input_length=input_tokens.size(-1),
        )

        outputs = model.generate(
            input_tokens,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            num_return_sequences=min(self.batch_size, num_samples),
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )

        gen_strs = self.tokenizer.batch_decode(
            outputs[:, input_tokens.size(-1) :],
            skip_special_tokens=self.skip_special_tokens,
        )
        outputs = []
        # removes eos tokens.
        for output in gen_strs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    min_index = min(min_index, output.index(eos))
            outputs.append(output[:min_index].replace("\t", "    "))
            
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs


class GenenralHfTorchDecoder(HfTorchDecoder):
    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.eos += ["\n```\n"]
        print(f"EOS strings: {self.eos}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)

    def codegen(
        self, 
        prompt: str, 
        dim: str,
        do_sample: bool = True, 
        num_samples: int = 200,
        strategy: str = 'customized',
        **kwargs
    ) -> List[str]:
        prompt = construct_prompt(prompt, strategy, dim, self.tokenizer, **kwargs)
        
        # TODO
        print('-' * 50)
        print(prompt)
        print('-' * 50)
        
        result = HfTorchDecoder.codegen(self, prompt, do_sample, num_samples)
        print(result)
        
        return result


class MistralChatDecoder(DecoderBase):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        kwargs = {}
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"
            kwargs["top_p"] = 0.95
            kwargs["temperature"] = self.temperature
        else:
            self.temperature = 0

        batch_size = min(self.batch_size, num_samples)

        outputs = []
        for _ in range(batch_size):
            ret = self.client.chat(
                model=self.name,
                messages=[
                    ChatMessage(
                        role="user",
                        content="Please generate code to solve the following problem in a Python markdown block:"
                        + f"\n```python\n{prompt.strip()}\n```",
                    )
                ],
                max_tokens=self.max_new_tokens,
                **kwargs,
            )

            outputs.append(ret.choices[0].message.content)

        return outputs

    def is_direct_completion(self) -> bool:
        return False


class AnthropicDecoder(DecoderBase, ABC):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_KEY"))

    def is_direct_completion(self) -> bool:
        return False


class AnthropicMessageDecoder(AnthropicDecoder):
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"

        batch_size = min(self.batch_size, num_samples)
        if not do_sample:
            assert batch_size == 1, "Sampling only supports batch size of 1"

        outputs = []
        for _ in range(batch_size):
            message = anthropic_request.make_auto_request(
                client=self.client,
                model=self.name,
                messages=[
                    {
                        "role": "user",
                        "content": "Please generate code to complete the following problem wrapped in a Python markdown block:"
                        + f"\n```python\n{prompt.strip()}\n```\n",
                    }
                ],
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                stop_sequences=["\n```\n", "\nif "],
            )
            outputs.append(message.content[0].text)

        return outputs


def make_model(
    model: str,
    backend: str,
    dataset: str,
    batch_size: int = 1,
    temperature: float = 0.0,
    tp=1,
    base_url=None,
    api_key=None,
):
    if backend == "vllm":
        return GeneralVllmDecoder(
            name=model,
            batch_size=batch_size,
            temperature=temperature,
            dataset=dataset,
            tp=tp,
        )
    elif backend == "hf":
        return GenenralHfTorchDecoder(
            name=model,
            batch_size=batch_size,
            temperature=temperature,
            dataset=dataset,
        )
    elif backend == "openai":
        return OpenAIChatDecoder(
            name=model,
            batch_size=batch_size,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key,
        )
    elif backend == "mistral":
        return MistralChatDecoder(
            name=model,
            batch_size=batch_size,
            temperature=temperature,
        )
    elif backend == "anthropic":
        return AnthropicMessageDecoder(
            name=model,
            batch_size=batch_size,
            temperature=temperature,
        )
    elif backend == "genai":
        return GenaiDecoder(
            name=model,
            batch_size=batch_size,
            temperature=temperature,
            api_key=api_key,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")
