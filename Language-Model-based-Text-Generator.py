import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2-large"
model = TFGPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Function to generate text based on user input
def generate_text(prompt, max_length=50, temperature=1.0, repetition_penalty=1.2):
    input_ids = tokenizer.encode(prompt, return_tensors="tf")
    attention_mask = tf.ones(input_ids.shape, dtype=tf.int32)
    
    # Generate text using the model
    output = model.generate(input_ids,
                            attention_mask=attention_mask,
                            pad_token_id=tokenizer.eos_token_id,
                            max_length=max_length,
                            temperature=temperature,
                            repetition_penalty=repetition_penalty,
                            num_return_sequences=1,
                            top_k=50,  # Experiment with this value
                            top_p=0.90)  # Experiment with this value
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


# Extension: Generating Multiple Variations
def generate_multiple_variations(prompt, num_variations=3):
    variations = []
    for i in range(num_variations):
        generated_variation = generate_text(prompt, repetition_penalty=1.0 + i * 0.2)
        variations.append(generated_variation)
    return variations



# Extension: User Control Over Text Style
def generate_text_with_style(prompt, style_tokens):
    input_ids = tokenizer.encode(prompt, return_tensors="tf")
    attention_mask = tf.ones(input_ids.shape, dtype=tf.int32)
    
    # Add style tokens to the input_ids
    input_ids = tf.concat([style_tokens, input_ids], axis=-1)
    
    # Generate text using the model
    output = model.generate(input_ids,
                            attention_mask=attention_mask,  # Set the attention mask
                            pad_token_id=tokenizer.eos_token_id,  # Set the pad token id
                            max_length=50)
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Extension: Interactive Conversations
def interactive_conversation():
    print("Bot: Hi there! I'm your interactive text generator.")
    while True:
        user_input = input("Bot: Enter a prompt.\nYou: ")
        user_style = input("Bot: Enter a style for your prompt (e.g., Happy, Mysterious, Surprising, etc.).\nYou: ")
        
        variations = generate_multiple_variations(user_input)
        print("Generated variations:")
        for i, variation in enumerate(variations, start=1):
            print(f"Variation {i}: {variation}")
        
        user_choice = input("Bot: Do you want to keep exploring or quit? (continue/exit)\nYou: ")
        if user_choice.lower() == "exit":
            print("Bot: Goodbye!")
            break

if __name__ == "__main__":
    print("Language Model-based Text Generator")
    interactive_conversation()
