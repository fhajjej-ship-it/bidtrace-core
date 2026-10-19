# SPDX-License-Identifier: MPL-2.0
"""Native GPU batches of isolated text prompts, each with its own grammar state."""
import time
import uuid


def generate_many(backend, jobs):
    from mlx_vlm.generate.ar import BatchGenerator
    from bidtrace_core.local.budget import build_budgeted_thinking_processor

    if not 1 <= len(jobs) <= 8:
        raise ValueError('Native local decoding accepts one to eight independent prompts')
    tokenizer = getattr(backend.processor, 'tokenizer', backend.processor)
    prompts, processors = [], []
    for messages, images, max_tokens, options in jobs:
        if images or options.get('enable_thinking') is not True or not options.get('response_schema'):
            raise ValueError('Native batching requires grounded text-only thinking contracts')
        prompt = backend._template(backend.processor, backend.config, messages,
                                   num_images=0, enable_thinking=True)
        if not prompt.endswith('<think>\n'):
            raise ValueError('Native prompt must preserve the original thinking prefill')
        prompts.append(backend._text_prompt_ids(prompt))
        processors.append([build_budgeted_thinking_processor(tokenizer,
            options['response_schema'], options['thinking_budget'])])
    batch_id = uuid.uuid4().hex
    started = time.perf_counter()
    generator = BatchGenerator(backend.model.language_model, backend.processor,
        completion_batch_size=len(jobs), prefill_batch_size=len(jobs),
        prefill_step_size=512, compute_logprobs=False)
    try:
        prompt_kwargs = [{key: value for key, value in backend.model.get_input_embeddings(
            backend.mx.array([ids]), None).to_dict().items() if value is not None} for ids in prompts]
        identifiers = generator.insert(prompts, [j[2] for j in jobs],
                                       prompt_kwargs=prompt_kwargs, logits_processors=processors)
        tokens = {identifier: [] for identifier in identifiers}
        finish = {}
        while generator.has_work:
            _, responses = generator.next()
            for response in responses:
                if response.finish_reason != 'stop':
                    tokens[response.uid].append(response.token)
                if response.finish_reason:
                    finish[response.uid] = response.finish_reason
        backend.mx.synchronize()
    finally:
        generator.close()
    elapsed = time.perf_counter() - started
    batch_stats = generator.stats()
    return [(tokenizer.decode(tokens[identifier], skip_special_tokens=False), {
        'prompt_tokens': len(prompts[index]), 'generation_tokens': len(tokens[identifier]) + 1,
        'total_tokens': len(prompts[index]) + len(tokens[identifier]) + 1,
        'cached_tokens': 0, 'finish_reason': finish.get(identifier), 'thinking_prefilled': True,
        'reasoning_tokens': processors[index][0].reasoning_tokens,
        'native_batch_id': batch_id, 'native_batch_size': len(jobs),
        'native_batch_seconds': round(elapsed, 6), 'seconds': round(elapsed / len(jobs), 6),
        'batch_prefill_seconds': batch_stats.prompt_time,
        'batch_generation_tps': batch_stats.generation_tps})
        for index, identifier in enumerate(identifiers)]
