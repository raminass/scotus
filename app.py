import gradio as gr
from transformers import pipeline
from utils import *
from datasets import load_dataset

pipe = pipeline(model="raminass/scotus-v10", top_k=13, padding=True, truncation=True)
all = load_dataset("raminass/full_opinions_1994_2020")
df = pd.DataFrame(all["train"])
choices = []
for index, row in df[df.category == "per_curiam"].iterrows():
    if len(row["text"]) > 1000:
        choices.append((f"""{row["case_name"]}""", [row["text"], row["year_filed"]]))

max_textboxes = 100


# https://www.gradio.app/guides/controlling-layout
def greet(opinion, year):
    judges_l = (
        df[(df["year_filed"] == year) & (df["category"] != "per_curiam")]
        .author_name.unique()
        .tolist()
    )

    if year == 1994:
        judges_l.extend(["Justice Breyer", "Justice Kennedy"])

    chunks = chunk_data(remove_citations(opinion))["text"].to_list()
    result = average_text(chunks, pipe, judges_l)
    k = len(chunks)

    wrt_boxes = []
    for i in range(k):
        wrt_boxes.append(gr.Textbox(chunks[i], visible=True))
        wrt_boxes.append(gr.Label(value=result[1][i], visible=True))
    return (
        [result[0]]
        + wrt_boxes
        + [gr.Textbox(visible=False), gr.Label(visible=False)] * (max_textboxes - k)
    )


def set_input(drop):
    return drop[0], drop[1], gr.Slider(visible=False)


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            opinion = gr.Textbox(label="Opinion")
            year = gr.Slider(1994, 2020, step=1, label="Year")
            drop = gr.Dropdown(choices=sorted(choices))
            with gr.Row():
                clear_btn = gr.Button("Clear")
                greet_btn = gr.Button("Predict")
        op_level = gr.outputs.Label(num_top_classes=13, label="Overall")

    textboxes = []
    for i in range(max_textboxes):
        with gr.Row():
            t = gr.Textbox(f"Textbox {i}", visible=False, label=f"Paragraph {i+1} Text")
            par_level = gr.Label(
                num_top_classes=5, label=f"Paragraph {i+1} Prediction", visible=False
            )
        textboxes.append(t)
        textboxes.append(par_level)

    drop.select(set_input, inputs=drop, outputs=[opinion, year, year])

    greet_btn.click(
        fn=greet,
        inputs=[opinion, year],
        outputs=[op_level] + textboxes,
    )

    clear_btn.click(
        fn=lambda: [None, 1994, gr.Slider(visible=True), None, None]
        + [gr.Textbox(visible=False), gr.Label(visible=False)] * max_textboxes,
        outputs=[opinion, year, year, drop, op_level] + textboxes,
    )


if __name__ == "__main__":
    demo.launch(debug=True)
