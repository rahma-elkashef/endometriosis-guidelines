import gradio as gr
import json

def rag_interface(question, pdf_file):
    # Case 1: User uploaded a medical report
    if pdf_file is not None:
        # Extract clinical data from the PDF using the existing process_medical_report function
        clinical_data = process_medical_report(pdf_file.name)

        # Formulate a query based on the extracted findings to ask the guidelines
        findings_text = ", ".join(clinical_data.get('findings', [])) if clinical_data.get('findings') else "endometriosis management"
        query = f"Based on these findings: {findings_text}, what are the NICE recommendations?"

        results = retrieve(query, top_k=5)
        answer = generate_answer(query, results)

        # Add the structured report summary to the evidence display
        evidence_md = "### 📋 Extracted Clinical Summary\n"
        evidence_md += "```json\n" + json.dumps(clinical_data, indent=2) + "\n```\n\n"
        evidence_md += "### 📚 Supporting Evidence from Guidelines\n"

    # Case 2: Standard text query
    else:
        query = question
        results = retrieve(query, top_k=5)
        answer = generate_answer(query, results)
        evidence_md = "### 📚 Supporting Evidence\n"

    # Format retrieval metadata
    for i, res in enumerate(results, start=1):
        source_info = (
            f"**[S{i}]** {res.get('guideline', 'NICE')} | "
            f"**Section:** {res.get('section_number', 'N/A')} ({res.get('section_title', 'N/A')}) | "
            f"**Page:** {res.get('page', 'N/A')}\n"
            f"*Chunk ID:* `{res.get('chunk_id', 'N/A')}` | "
            f"*Retrieval Score:* `{res.get('rerank_score', 0):.4f}`"
        )
        evidence_md += f"\n---\n{source_info}\n\n*\"{res.get('text', '')}\"*\n"

    return answer, evidence_md

# Create the Updated Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🩺 NICE Endometriosis Decision Support")
    gr.Markdown("Ask a clinical question or upload a medical report PDF for grounded analysis.")

    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="User Question",
                placeholder="e.g., When should laparoscopy be considered?",
                lines=3
            )
            file_upload = gr.File(
                label="Upload Medical Report (PDF)",
                file_types=[".pdf"]
            )
            submit_btn = gr.Button("Ask Guidelines / Analyze Report", variant="primary")
            clear_btn = gr.Button("Clear")

        with gr.Column(scale=2):
            output_answer = gr.Markdown(label="LLM Recommendation")
            output_evidence = gr.Markdown(label="Source Context & Clinical Data")

    submit_btn.click(
        fn=rag_interface,
        inputs=[input_text, file_upload],
        outputs=[output_answer, output_evidence]
    )

    clear_btn.click(lambda: [None, None, None, None], None, [input_text, file_upload, output_answer, output_evidence])

demo.launch(quiet=True)
