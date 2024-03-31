import gradio as gr
import time

from rag_components import RetrievalAugmentedGenerator


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "HuggingFaceH4/zephyr-7b-beta"  # https://huggingface.co/HuggingFaceH4/zephyr-7b-beta
rag = RetrievalAugmentedGenerator(embed_model_name=EMBED_MODEL, gen_model_name=GEN_MODEL)


def add_message(history: list, message: str) -> (list, gr.MultimodalTextbox):
    """
    Upon submission to the text box, add the submitted message to the chat history.

    :param history: List of previously submitted messages.
    :param message: New message string input from the user.

    :return: History, and a MultimodalTextbox with no value that is not interactive
        (it will prevent the user from submitting while the bot's response is still
        being retrieved).
    """
    if len(message["files"]) > 0:
        # Run the ingest pipeline for the uploaded documents
        file_paths = [x["path"] for x in message["files"]]
        rag.ingest_documents(docs=file_paths)

        # Append submitted files
        for x in message["files"]:
            history.append(((x["path"],), None))

    # Append message text
    if message["text"] is not None and message["text"] != "":
        history.append((message["text"], None))

    return (
        history,
        gr.MultimodalTextbox(value=None, interactive=False, file_types=["text"])
    )


def get_bot_response(history: list):
    """A generator for chat history."""
    # Get the LLM's response
    if ".txt" == history[-1][0][0][-4:]:  # this is hacky, but whatever
        response = "Thank you, I will refer to the provided document for future responses."
    else:
        response = rag.retrieve_and_generate(query=history[-1][0])

    # Create a placeholder for a new history entry so that the response will
    # 'stream' to it.
    history[-1][1] = ""

    # Stream the response, character by character, and append to the new message in the
    # chat history.
    for character in response:
        history[-1][1] += character
        time.sleep(0.005)
        yield history


with gr.Blocks() as demo:

    chatbot = gr.Chatbot(
        value=[],  # this will hold the chat history
        elem_id="chatbot",
        label="Chat History",
        bubble_full_width=False,
    )

    chat_input = gr.MultimodalTextbox(
        interactive=True,
        file_types=["text"],
        placeholder="Enter a message or upload a .txt file to discuss...",
        show_label=False,
        submit_btn=">",
    )

    # Specify what to do upon submission to chat_input
    # First update the chatbot component with the user message,
    # then get the bot's response and append it to the chat history,
    # and then display the bot's response.
    chat_msg = chat_input.submit(
        fn=add_message,
        inputs=[chatbot, chat_input],
        outputs=[chatbot, chat_input],
        queue=False
    ).then(
        fn=get_bot_response,
        inputs=chatbot,
        outputs=chatbot,
        api_name="bot_response"
    ).then(
        fn=lambda: gr.Textbox(interactive=True),
        inputs=None,
        outputs=[chat_input],
        queue=False
    )

    # A button to clear chat history
    clear = gr.ClearButton(components=[chat_input, chatbot], value="Clear Chat History")

demo.queue()


if __name__ == "__main__":
    demo.launch()
