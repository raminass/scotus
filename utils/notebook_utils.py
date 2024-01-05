from IPython.display import display, HTML
from sklearn.model_selection import train_test_split


def find_case_by_name(df, name):
    names_list = df["case_name"].str.lower()
    return display(
        HTML(
            df[names_list.str.contains(name.lower())]
            .iloc[:, :-1]
            .to_html(render_links=True, escape=False)
        )
    )
