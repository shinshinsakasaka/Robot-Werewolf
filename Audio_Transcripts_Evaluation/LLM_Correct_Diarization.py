
import pandas as pd
import json
import time
from openai import AzureOpenAI

# Initialize Azure OpenAI client (replace with your actual details)
client = AzureOpenAI(
    api_key="",
    api_version="2024-02-01",
    azure_endpoint="https://bl-info-werewolfrobot.openai.azure.com/" # <<< IMPORTANT: Replace with your actual endpoint
)

# Load your CSV
try:
    df = pd.read_csv("/Users/shinsaka/Desktop/Python/Werewolf-robot/Data/Baseline_Test_Data_Audio/test_transcript_output_group_6_experiment_4.csv")
    print(f"DataFrame loaded. Head:\n{df.head()}")
    print(f"DataFrame columns: {df.columns.tolist()}")
except FileNotFoundError:
    print("Error: 'your_conversation.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# --- SYSTEM MESSAGE HERE ---
system_message = """You are an expert transcriber and conversational analyst. Your task is to review segments of a conversation and correct any errors in speaker diarization based on the flow of dialogue, logical consistency, and common conversational patterns.

You MUST output a JSON object. This JSON object MUST contain a single key, "segments", whose value is a JSON array. This JSON array ("segments") MUST contain *all* the conversation segments provided in the input, even if no speaker change is made for a segment. For each segment in the "segments" array, you MUST retain the original 'start_time' and 'end_time' values exactly as provided. ONLY change the 'speaker' value if a correction is needed.

Each object within the "segments" array MUST have 'start_time', 'end_time', 'speaker', and 'text' keys. Do NOT include any other text or explanation outside the JSON object."""

corrected_segments = []
chunk_size = 20 # Number of conversation rows to send at once
overlap_size = 5 # Number of rows to overlap for context

for i in range(0, len(df), chunk_size - overlap_size):
    print(f"\n--- Processing chunk starting at loop index: {i} ---")

    start_idx = i
    end_idx = min(i + chunk_size, len(df))
    
    if i > 0:
        start_idx = max(0, i - overlap_size) 

    current_chunk_df = df.iloc[start_idx:end_idx].copy() # Use .copy() to avoid SettingWithCopyWarning

    conversation_text = ""
    if current_chunk_df.empty:
        print("Warning: current_chunk_df is empty. Skipping this chunk.")
        continue

    for idx, row in current_chunk_df.iterrows():
        # Ensure column names match your CSV exactly (case-sensitive)
        conversation_text += f"[{row['start_time']}-{row['end_time']}][Speaker:{row['speaker']}] {row['text']}\n"

    # --- USER MESSAGE CONTENT HERE (within the loop, as it changes per chunk) ---
    user_message_content = f"""Here is a segment of a conversation. Each line starts with timestamps and the current speaker label. Your task is to review the speaker labels and correct them if they appear to be wrong based on the conversational context.

    ---
    **Example:**

    Input Conversation:
    [0:00:00-0:00:02][Speaker:A] Hello there.
    [0:00:02-0:00:03][Speaker:A] How are you?
    [0:00:03-0:00:05][Speaker:B] I'm fine, thank you.
    [0:00:05-0:00:06][Speaker:A] Great!

    Expected Output JSON:
    {{
        "segments": [
            {{"start_time": "0:00:00", "end_time": "0:00:02", "speaker": "SPEAKER_01", "text": "Hello there."}},
            {{"start_time": "0:00:02", "end_time": "0:00:03", "speaker": "SPEAKER_01", "text": "How are you?"}},
            {{"start_time": "0:00:03", "end_time": "0:00:05", "speaker": "SPEAKER_00", "text": "I'm fine, thank you."}},
            {{"start_time": "0:00:05", "end_time": "0:00:06", "speaker": "SPEAKER_01", "text": "Great."}}
        ]
    }}

    ---

    Now, please process the following Conversation Segment. Return the corrected segments in the exact JSON object format shown in the example above, retaining all original timestamps, under the "segments" key:

    Conversation Segment:
    {conversation_text}

    Output the corrected segment in JSON object format:
    """

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message_content}
    ]

    try:
        print(f"Sending request to OpenAI for chunk starting at index {start_idx}...")
        response = client.chat.completions.create(
            model="gpt-4", # or your specific deployment name, e.g., "my-gpt4-deployment"
            messages=messages,
            response_format={"type": "json_object"} # We ask for a JSON object. Model should respond with an array.
        )
        print("Response received from OpenAI.")
        
        raw_llm_output = response.choices[0].message.content
        print(f"\n--- Raw LLM output for chunk starting at {start_idx} ---")
        print(raw_llm_output) 
        print(f"--- End Raw LLM output ---\n")

        # 1. Parse the full JSON object returned by the LLM
        full_llm_response_object = json.loads(raw_llm_output)
        
        # 2. Validate that it's a dictionary and contains the 'segments' key
        if not isinstance(full_llm_response_object, dict) or "segments" not in full_llm_response_object:
            raise TypeError(f"LLM response did not contain expected root object or 'segments' key. Content: {raw_llm_output}")
        
        # 3. EXTRACT the actual list of segments from the 'segments' key
        corrected_data = full_llm_response_object["segments"] 

        # 4. Now, validate that the extracted 'segments' is indeed a list
        if not isinstance(corrected_data, list):
            raise TypeError(f"The 'segments' value is not a list as expected. Type: {type(corrected_data)}. Content: {raw_llm_output}")

        print(f"Type of parsed corrected_data (extracted segments): {type(corrected_data)}")
        print(f"Length of parsed corrected_data (extracted segments): {len(corrected_data)}")
        if len(corrected_data) > 0:
            print(f"First item of parsed corrected_data: {corrected_data[0]}")
        else:
            print("WARNING: Extracted 'segments' list is EMPTY! This will cause 'No match found' for all rows in the chunk.")

        # Merge corrected data back into your original structure.
        for original_idx, original_row in current_chunk_df.iterrows():
            found_match = False
            for corrected_item in corrected_data:
                # Basic validation for corrected_item
                if not isinstance(corrected_item, dict):
                    print(f"  ERROR: corrected_item is not a dictionary. Type: {type(corrected_item)}. Content: {corrected_item}")
                    continue # Skip this malformed item

                required_keys = ['start_time', 'end_time', 'speaker', 'text']
                if not all(key in corrected_item for key in required_keys):
                     print(f"  ERROR: Corrected item missing required keys ({required_keys}): {corrected_item.keys()}")
                     continue # Skip this item if keys are missing

                # Compare as strings to avoid type mismatch issues (e.g., if one is float, other is string)
                # print(f"  Comparing original '{original_row['start_time']}' == corrected '{corrected_item['start_time']}' and "
                #       f"original '{original_row['end_time']}' == corrected '{corrected_item['end_time']}'") # Uncomment for detailed match debug
                if str(original_row['start_time']) == str(corrected_item['start_time']) and \
                   str(original_row['end_time']) == str(corrected_item['end_time']):
                    
                    df.loc[original_idx, 'speaker'] = corrected_item['speaker']
                    found_match = True
                    break 
            
            if not found_match:
                print(f"Warning: No match found in LLM output for original row at index {original_idx} "
                      f"(start_time={original_row['start_time']}, end_time={original_row['end_time']}). "
                      "This row's speaker will not be updated by the LLM for this chunk.")


        time.sleep(1)

    except json.JSONDecodeError as e:
        print(f"JSON Decoding Error for chunk starting at index {start_idx}: {e}")
        print(f"Problematic LLM Output: {raw_llm_output}")
        continue
    except Exception as e:
        print(f"Generic Error processing chunk starting at index {start_idx}: {e}")
        import traceback
        traceback.print_exc()
        continue

df.to_csv("corrected_conversation.csv", index=False)
print("Diarization correction complete. Saved to corrected_conversation.csv")
