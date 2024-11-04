from transformers import MistralForCausalLM


class CustomMistral(MistralForCausalLM):
    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        labels = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        cache_position = None,
        dataset_index = None,
    ):
        return super().forward(input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict, cache_position)
