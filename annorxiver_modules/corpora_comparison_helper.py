from collections import Counter
import csv
import string

from lxml import etree
from mizani.formatters import custom_format, log_format
import numpy as np
import pandas as pd
import plotnine as p9
from scipy.stats import norm
import spacy
import svgutils.transform as sg
from svgutils.compose import Unit
from tqdm import tqdm_notebook

from annorxiver_modules.document_helper import dump_article_text


def aggregate_word_counts(doc_iterator, disable_progressbar=False):
    """
    This function aggregates the word count tsv files.
    Arguments:
        doc_iterator - a pathlib generator that returns file paths to be parsed
        disable_progressbar - show the progressbar?
    Example tsv file:
    | lemma | pos_tag | dep_tag | count |
    | --- | --- | --- | --- |
    | genome | NOUN | compound | 23 |
    ...
    """
    global_word_counter = Counter()

    for doc in tqdm_notebook(doc_iterator, disable=disable_progressbar):
        with open(doc, "r") as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter="\t")

            # create keys with the following types
            # lemma - the token string
            # pos_tag - the part of speech tag
            # dep_tag - the dependency path tag
            global_word_counter.update(
                {
                    (row["lemma"], row["pos_tag"], row["dep_tag"]): int(row["count"])
                    for row in reader
                }
            )

    return global_word_counter


def calculate_confidence_intervals(data_df):
    """
    Calculates the 95% confidence intervals for the odds ratio
    confidence intervals calculated from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2938757/
    p-values obtained from https://www.bmj.com/content/343/bmj.d2304 except using scipy.stats.norm.sf function *2
    (two-sided)
    Arguments:
        data_df - the dataframe used to calculate the bars
    """
    # 95% confidence
    crit_value = 1.96
    ci_df = data_df.assign(
        lower_odds=lambda x: np.exp(
            np.log(x.odds_ratio)
            - 1.96
            * (
                # log(odds) - z_alpha/2*sqrt(1/a+1/b+1/c+1/d)
                np.sqrt(
                    1 / x.corpus_one_a
                    + 1 / x.corpus_two_b
                    + 1 / x.corpus_one_c
                    + 1 / x.corpus_two_d
                )
            )
        ),
        upper_odds=lambda x: np.exp(
            np.log(x.odds_ratio)
            + 1.96
            * (
                # log(odds)+ z_alpha/2*sqrt(1/a+1/b+1/c+1/d)
                np.sqrt(
                    1 / x.corpus_one_a
                    + 1 / x.corpus_two_b
                    + 1 / x.corpus_one_c
                    + 1 / x.corpus_two_d
                )
            )
        ),
    ).assign(
        z_value=lambda x: (
            np.log(x.odds_ratio)
            / ((np.log(x.upper_odds) - np.log(x.lower_odds)) / (2 * crit_value))
        ),
        p_value=lambda x: norm.sf(
            np.abs(
                np.log(x.odds_ratio)
                / ((np.log(x.upper_odds) - np.log(x.lower_odds)) / (2 * crit_value))
            )
        ),
    )
    return ci_df


def create_lemma_count_df(data_df, corpus_one_label, corpus_two_label):
    """
    Creates a dataframe that contains lemmas (tokens in base form)and their respective counts
    Arguments:
        data_df - the dataframe to convert
        corpus_one_label - the label for the first corpus
        corpus_two_label - the label for the second corpus
    """

    count_plot_df = pd.DataFrame(
        list(
            zip(
                data_df.lemma.tolist(),
                data_df.corpus_one_a.tolist(),
                data_df.assign(label=corpus_one_label).label.tolist(),
            )
        )
        + list(
            zip(
                data_df.lemma.tolist(),
                data_df.corpus_two_b.tolist(),
                data_df.assign(label=corpus_two_label).label.tolist(),
            )
        ),
        columns=["lemma", "count", "repository"],
    )
    return count_plot_df


def dump_to_dataframe(count_dict, file_name):
    """
    This function outputs the counter data object in
    the form of a tsv file. This was created to avoid
    the memory issue PMC creates.
    Arguments:
        count_dict - the counter object to be outputted
        file_name - the name of the file for the output to be written to
    """

    # Avoids memory issues especially with the PMC.
    # A dictionary form of PMC tokens cannot be picked
    # without ram > 64GB.
    # This solution prevents crashes and is quite convenient to
    # work with.
    with open(file_name, "w") as tsvfile:
        reader = csv.DictWriter(
            tsvfile, fieldnames=["lemma", "pos_tag", "dep_tag", "count"], delimiter="\t"
        )

        reader.writeheader()
        for item in tqdm_notebook(count_dict):
            reader.writerow(
                {
                    "lemma": item[0],
                    "pos_tag": item[1],
                    "dep_tag": item[2],
                    "count": count_dict[item],
                }
            )


def get_term_statistics(
    corpus_one, corpus_two, freq_num, psudeocount=1, disable_progressbar=False
):
    """
    This function is designed to perform the folllowing calculations:
        - log likelihood of contingency table
        - log odds ratio
    keywords:
        corpus_one - a dataframe object with terms and counts
        corpus_two - a datafram object with terms and counts
        freq_num - number of most common words to use from both corpora
        psudeocount - the psudocount to avoid divide by zero
        disable_progressbar - show the progress bar?
    """

    spacy_nlp = spacy.load("en_core_web_sm")
    stop_word_list = list(spacy_nlp.Defaults.stop_words)

    # Remove special characters here when calculating odds ratio
    term_list = set(
        corpus_one.query("lemma.str.len() > 1")
        .query("lemma.str.contains(r'[a-z]')")
        .query(f"lemma not in {stop_word_list}")
        .sort_values("count", ascending=False)
        .head(freq_num)
        .lemma.values
    ) | set(
        corpus_two.query("lemma.str.len() > 1")
        .query("lemma.str.contains(r'[a-z]')")
        .query(f"lemma not in {stop_word_list}")
        .sort_values("count", ascending=False)
        .head(freq_num)
        .lemma.values
    )

    corpus_one_total = corpus_one["count"].sum()
    corpus_two_total = corpus_two["count"].sum()

    term_data = []
    for term in tqdm_notebook(term_list, disable=disable_progressbar):

        corpus_one_term_count = (
            corpus_one.query(f"lemma=={repr(term)}")["count"].values[0]
            if term in corpus_one.lemma.tolist()
            else 0
        )

        corpus_two_term_count = (
            corpus_two.query(f"lemma=={repr(term)}")["count"].values[0]
            if term in corpus_two.lemma.tolist()
            else 0
        )

        observed_contingency_table = np.array(
            [
                [corpus_one_term_count, corpus_two_term_count],
                [corpus_one_total, corpus_two_total],
            ]
        )

        # Log Likelihood

        ## add psudeocount to prevent log(0)
        observed_contingency_table += psudeocount

        a, b, c, d = (
            observed_contingency_table[0][0],
            observed_contingency_table[0][1],
            observed_contingency_table[1][0],
            observed_contingency_table[1][1],
        )

        # Obtained from (Kilgarriff, 2001) - Comparing Corpora
        def LL(a, b, c, d):
            return 2 * (
                a * np.log(a)
                + b * np.log(b)
                + c * np.log(c)
                + d * np.log(d)
                - (a + b) * np.log(a + b)
                - (a + c) * np.log(a + c)
                - (b + d) * np.log(b + d)
                - (c + d) * np.log(c + d)
                + (a + b + c + d) * np.log(a + b + c + d)
            )

        log_likelihood = LL(a, b, c, d)

        # Log Odds
        log_ratio = float((a * d) / (b * c))

        term_data.append(
            {
                "lemma": term,
                "corpus_one_a": a,
                "corpus_two_b": b,
                "corpus_one_c": c,
                "corpus_two_d": d,
                "log_likelihood": log_likelihood,
                "odds_ratio": log_ratio,
            }
        )

    return pd.DataFrame.from_records(term_data)


def get_word_stats(
    document_list,
    document_folder,
    tag_path,
    output_folder,
    skip_condition=lambda folder, document: False,
):
    """
    This function calculates the token statistics for each corpus
    Arguments:
        document_list - the list of document paths
        document_folder - the folder to look for each document
        tag_path - the xpath used to parse the xml files
        output_folder - the folder to contain individual file stats
        skip_condition - a lambda function that skips files based on user input
    """

    sentence_length = {}
    lemma_model = spacy.load("en_core_web_sm")
    lemma_model.max_length = 9000000

    for document in tqdm_notebook(document_list):

        if skip_condition(document_folder, document):
            continue

        document_text = dump_article_text(
            file_path=f"{document_folder}/{document}",
            xpath_str=tag_path,
            remove_stop_words=False,
        )

        doc = lemma_model(" ".join(document_text), disable=["ner"])

        tokens = [
            (str(tok).lower(), tok.pos_, tok.dep_)
            for tok in doc
            if tok.text.lower() not in string.punctuation
        ]

        sentence_length[document] = [len(sent) for sent in doc.sents]

        with open(f"{output_folder}/{document.stem}.tsv", "w") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["lemma", "pos_tag", "dep_tag", "count"],
                delimiter="\t",
            )

            writer.writeheader()

            lemma_stats = Counter(tokens)
            writer.writerows(
                [
                    {
                        "lemma": val[0][0],
                        "pos_tag": val[0][1],
                        "dep_tag": val[0][2],
                        "count": val[1],
                    }
                    for val in lemma_stats.items()
                ]
            )

    return sentence_length


def KL_divergence(corpus_top, corpus_bottom, num_terms=100, pseudo_count=1):
    """
    This function calculates the KL divergence between two corpora
    KL(D||P)
    Arguments:
        corpus_top - the D in the above equation
        corpus_bottom - the P in the above equation
        num_terms - the number of terms used to evaluate KL
        pseudo_count - the number to add incase a term is missing from the corpus' vocab
    """
    term_list = list(
        set(
            corpus_top.sort_values("count", ascending=False)
            .head(num_terms)
            .lemma.values
        )
        | set(
            corpus_bottom.sort_values("count", ascending=False)
            .head(num_terms)
            .lemma.values
        )
    )
    corpus_top_total = corpus_top["count"].sum()
    corpus_bottom_total = corpus_bottom["count"].sum()

    top_term_freq = (
        corpus_top.query(f"lemma in {term_list}")
        .assign(freq=lambda x: x["count"] / corpus_top_total)
        .sort_values("lemma")
    )

    bottom_term_freq = (
        corpus_bottom.query(f"lemma in {term_list}")
        .assign(freq=lambda x: x["count"] / corpus_bottom_total)
        .sort_values("lemma")
    )

    if top_term_freq.shape[0] != bottom_term_freq.shape[0]:

        missing_terms = set(term_list) - set(bottom_term_freq.lemma.tolist())
        if len(missing_terms) > 0:
            bottom_term_freq = bottom_term_freq.append(
                pd.DataFrame(missing_terms, columns=["lemma"]).assign(
                    count=pseudo_count, freq=pseudo_count / corpus_bottom_total
                ),
            )

        missing_terms = set(term_list) - set(top_term_freq.lemma.tolist())
        if len(missing_terms) > 0:
            top_term_freq = top_term_freq.append(
                pd.DataFrame(missing_terms, columns=["lemma"]).assign(
                    count=pseudo_count, freq=pseudo_count / corpus_top_total
                ),
            )

    assert top_term_freq.shape[0] == bottom_term_freq.shape[0]

    kl_div = top_term_freq.freq.values * np.log(
        top_term_freq.freq.values / bottom_term_freq.freq.values
    )

    return kl_div.sum()


def plot_bargraph(count_plot_df, plot_df):
    """
    Plots the bargraph
    Arguments:
        count_plot_df - The dataframe that contains lemma counts
        plot_df - the dataframe that contains the odds ratio and lemmas
    """

    graph = (
        p9.ggplot(count_plot_df.astype({"count": int}), p9.aes(x="lemma", y="count"))
        + p9.geom_col(position=p9.position_dodge(width=0.5), fill="#253494")
        + p9.coord_flip()
        + p9.facet_wrap("repository", scales="free_x", dir="h")
        + p9.scale_x_discrete(
            limits=(plot_df.sort_values("odds_ratio", ascending=True).lemma.tolist())
        )
        + p9.scale_y_continuous(labels=custom_format("{:,.0g}"))
        + p9.labs(x=None)
        + p9.theme_seaborn(context="paper", style="ticks", font="Arial", font_scale=1.8)
        + p9.theme(
            figure_size=(11, 8.5),
            strip_background=p9.element_rect(fill="white"),
            strip_text=p9.element_text(size=14),
            axis_title=p9.element_text(size=14),
            axis_text_x=p9.element_text(size=12),
        )
    )
    return graph


def plot_point_bar_figure(figure_one_path, figure_two_path):
    """
    Combines the pointplot and bargraph together using svg magic
    Arguments:
        figure_one_path - The pointplot figure
        figure_two_path - The barplot figure
    """
    fig1 = sg.fromfile(figure_one_path)
    fig2 = sg.fromfile(figure_two_path)

    fig1_width_size = np.round(float(fig1.root.attrib["width"][:-2]) * 1.33, 0)
    fig1_height_size = np.round(float(fig1.root.attrib["height"][:-2]) * 1.33, 0)

    fig2_width_size = np.round(float(fig2.root.attrib["width"][:-2]) * 1.33, 0)
    fig2_height_size = np.round(float(fig2.root.attrib["height"][:-2]) * 1.33, 0)

    fig = sg.SVGFigure(
        Unit((fig1_width_size + fig2_width_size) - 360),
        Unit(min(fig1_height_size, fig2_height_size) - 50),
    )

    fig.append(
        [etree.Element("rect", {"width": "100%", "height": "100%", "fill": "white"})]
    )

    plot1 = fig1.getroot()
    plot1.moveto(10, 30)

    plot2 = fig2.getroot()
    plot2.moveto(fig1_width_size - 160, 12)

    text_A = sg.TextElement(10, 30, "A", size=18, weight="bold")
    text_B = sg.TextElement(fig1_width_size - 160, 30, "B", size=18, weight="bold")

    fig.append([plot1, plot2, text_A, text_B])

    return fig
