import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402
from models import (  # noqa: E402
    ALL_MODELS,
    BASELINE,
    BILSTM,
    CHOICES,
    GEMMA,
    REPO_ROOT,
    gemma_status,
    run,
)


def load_examples() -> list[list[str]]:
    path = REPO_ROOT / "dataset" / "test.csv"
    if not path.exists():
        return []

    df = pd.read_csv(path).sample(10)
    return [
        [str(row["prompt"])] + [str(row[c]) for c in CHOICES]
        for _, row in df.iterrows()
    ]


def solve(model_name: str, question: str, *options: str):
    if not question.strip():
        return "Enter a question first.", pd.DataFrame(), "", ""

    if not all(o.strip() for o in options):
        return "Fill in all five options.", pd.DataFrame(), "", ""

    prediction = run(model_name, question, list(options))
    if prediction.note:
        unavailable = "Model unavailable"
        return (
            f"**Unavailable.** {prediction.note}",
            pd.DataFrame(),
            unavailable,
            unavailable
        )

    table = pd.DataFrame(
        {
            "Option": CHOICES,
            "Probability": [f"{p:.4f}" for p in prediction.probabilities],
            "Rank": [prediction.ranking.index(c) + 1 for c in CHOICES],
        }
    ).sort_values("Rank")

    winner = prediction.ranking[0]
    answer = (
        f"### Prediction: `{prediction.top3}`\n"
        f"Top choice is **{winner}** with "
        f"{prediction.probabilities[CHOICES.index(winner)]:.1%} confidence.\n\n"
        f"*Submitted to Kaggle as three letters in this order, scored by MAP@3.*"
    )
    none = "This model does not use retrieval."
    return answer, table, prediction.context or none, prediction.model_context or none


def build_ui() -> gr.Blocks:
    examples = load_examples()
    unavailable = gemma_status()

    with gr.Blocks(title="Smart MCQ Solver") as demo:
        gr.Markdown(
            "# Smart MCQ Solver\n"
            "Three models for the Kaggle Smart MCQ Solver Challenge: a BiLSTM "
            "written from scratch, a LoRA fine-tuned Gemma-4, and a classical "
            "TF-IDF baseline. Enter a question with five options, or load an "
            "example below.\n\n"
            "*Varun Agnihotri (24f2004142) - "
            "[GitHub](https://github.com/PythonicVarun/GenAI_Project-May_2026)*"
        )
        if unavailable:
            gr.Markdown(f"> **Note on {GEMMA}:** {unavailable}")

        with gr.Row():
            with gr.Column(scale=3):
                question = gr.Textbox(
                    label="Question", lines=2, placeholder="What is ...?"
                )
                options = [gr.Textbox(label=f"Option {c}", lines=3) for c in CHOICES]
            with gr.Column(scale=2):
                model_name = gr.Radio(
                    ALL_MODELS,
                    value=BILSTM,
                    label="Model",
                    info=f"{BASELINE} is the fastest; {BILSTM} scores best.",
                )
                submit = gr.Button("Solve", variant="primary")
                answer = gr.Markdown()
                table = gr.Dataframe(
                    headers=["Option", "Probability", "Rank"],
                    label="Per-option probabilities",
                    interactive=False,
                    visible=True,
                )

        with gr.Accordion("Retrieved context (the RAG step)", open=False):
            gr.Markdown(
                "Both deep models prepend similar solved questions from the "
                "training set to the prompt. Left is what the retriever picked, "
                "in the original wording. Right is the same text after "
                "`clean()` lowercases it and strips punctuation, which is what "
                "the BiLSTM is actually tokenised from - its vocabulary was "
                "built the same way."
            )
            with gr.Row():
                context = gr.Textbox(label="Retrieved rows (original)", lines=10)
                model_context = gr.Textbox(
                    label="As the model receives it (cleaned)", lines=10
                )

        submit.click(
            solve,
            inputs=[model_name, question, *options],
            outputs=[answer, table, context, model_context],
        )
        model_name.change(
            lambda name: gr.update(visible=name != GEMMA),
            inputs=model_name,
            outputs=table,
        )

        if examples:
            gr.Examples(
                examples=examples,
                inputs=[question, *options],
                label="Examples from the test set",
            )

    return demo


def start():
    build_ui().launch(theme=gr.themes.Soft())


if __name__ == "__main__":
    start()
