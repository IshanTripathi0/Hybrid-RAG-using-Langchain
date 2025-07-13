import gradio as gr
from rag_pipeline import load_rag_pipeline

# Load the pipeline once
print("Initializing RAG pipeline...")
rag_chain = load_rag_pipeline()
print("RAG pipeline ready.")

def rag_query_interface(query: str):
    if not query.strip():
        return "Please enter a question.", ""
    
    try:
        # Run RAG pipeline
        result = rag_chain.invoke(query)
        return result, ""  # or include context if you modify the pipeline
    except Exception as e:
        return f"❌ Error: {str(e)}", ""

# Launch Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("##  Hybrid RAG Q&A for IMDb Reviews")
    gr.Markdown("Ask questions about movie reviews using hybrid BM25 + embedding retrieval.")

    with gr.Row():
        input_box = gr.Textbox(label="Your Question", placeholder="e.g. A movie about rebellion and identity", lines=1)
    
    with gr.Row():
        output_box = gr.Textbox(label="Answer", lines=5)
    
    with gr.Row():
        btn = gr.Button("Submit")

    btn.click(fn=rag_query_interface, inputs=input_box, outputs=output_box)

# Run the app
if __name__ == "__main__":
    demo.launch()

