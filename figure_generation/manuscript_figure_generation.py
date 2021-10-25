# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1+dev
#   kernelspec:
#     display_name: Python [conda env:annorxiver]
#     language: python
#     name: conda-env-annorxiver-py
# ---

# # Figure Generation for Manuscript

# +
import os
from pathlib import Path
from IPython.display import Image, display, SVG

from cairosvg import svg2png
from lxml import etree
import numpy as np
from svgutils.compose import Unit
import svgutils.transform as sg

from annorxiver_modules.pca_plot_helper import load_clouds
# -

# ## Figure 1

corpora_comparison_path = Path("../biorxiv/corpora_comparison/output/")

# +
panel_one = sg.fromfile(corpora_comparison_path / "svg_files/corpora_kl_divergence.svg")

# Convert pt units to pixel units
# Vince's tutorial FTW
panel_one_size = (
    np.round(float(panel_one.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_one.root.attrib["height"][:-2]) * 1.33, 0),
)

scale_x = 1.4
scale_y = 1.4

print(f"original: {panel_one_size}")
print(f"scaled:{(panel_one_size[0]*scale_x,panel_one_size[1]*scale_y)}")

panel_one = panel_one.getroot()
panel_one.scale(x=scale_x, y=scale_y)
panel_one.moveto(50, 30)

# +
panel_two = sg.fromfile(
    corpora_comparison_path / "svg_files/biorxiv_pmc_frequency_odds.svg"
)

panel_two_size = (
    np.round(float(panel_two.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_two.root.attrib["height"][:-2]) * 1.33, 0),
)
scale_x = 1.4
scale_y = 1.4

print(f"original: {panel_two_size}")
print(f"scaled:{(panel_two_size[0]*scale_x, panel_two_size[1]*scale_y)}")

panel_two = panel_two.getroot()
panel_two.scale(x=scale_x, y=scale_y)
panel_two.moveto(10, 820)

# +
panel_three = sg.fromfile(
    corpora_comparison_path / "svg_files/biorxiv_pmc_frequency_bar.svg"
)

panel_three_size = (
    np.round(float(panel_three.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_three.root.attrib["height"][:-2]) * 1.33, 0),
)
scale_x = 1.4
scale_y = 1.4

print(f"original: {panel_three_size}")
print(f"scaled:{(panel_three_size[0]*scale_x, panel_three_size[1]*scale_y)}")

panel_three = panel_three.getroot()
panel_three.scale(x=scale_x, y=scale_y)
panel_three.moveto(1100, 790)

# +
panel_four = sg.fromfile(
    corpora_comparison_path / "svg_files/preprint_published_frequency_odds.svg"
)

panel_four_size = (
    np.round(float(panel_four.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_four.root.attrib["height"][:-2]) * 1.33, 0),
)
scale_x = 1.4
scale_y = 1.4
print(f"original: {panel_four_size}")
print(f"scaled:{(panel_four_size[0]*scale_x, panel_four_size[1]*scale_y)}")

panel_four = panel_four.getroot()
panel_four.scale(x=scale_x, y=scale_y)
panel_four.moveto(10, 1690)

# +
panel_five = sg.fromfile(
    corpora_comparison_path / "svg_files/preprint_published_frequency_bar.svg"
)

panel_five_size = (
    np.round(float(panel_five.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_five.root.attrib["height"][:-2]) * 1.33, 0),
)
scale_x = 1.4
scale_y = 1.4

print(f"original: {panel_five_size}")
print(f"scaled:{(panel_five_size[0]*scale_x, panel_five_size[1]*scale_y)}")

panel_five = panel_five.getroot()
panel_five.scale(x=scale_x, y=scale_y)
panel_five.moveto(1070, 1670)
# -

panel_one_label = sg.TextElement(30, 30, "A", size=30, weight="bold")
panel_two_label = sg.TextElement(10, 800, "B", size=30, weight="bold")
panel_three_label = sg.TextElement(1100, 800, "C", size=30, weight="bold")
panel_four_label = sg.TextElement(30, 1670, "D", size=30, weight="bold")
panel_five_label = sg.TextElement(1100, 1670, "E", size=30, weight="bold")

# +
figure_one = sg.SVGFigure(Unit(2127), Unit(2433))

figure_one.append(
    [
        etree.Element("rect", {"width": "100%", "height": "100%", "fill": "white"}),
        panel_one,
        panel_two,
        panel_three,
        panel_four,
        panel_five,
        panel_one_label,
        panel_two_label,
        panel_three_label,
        panel_four_label,
        panel_five_label,
    ]
)
display(SVG(figure_one.to_str()))
# -

# save generated SVG files
figure_one.save("output/Figure_1.svg")
svg2png(bytestring=figure_one.to_str(), write_to="output/Figure_1.png", dpi=600)

os.system(
    "convert -compress LZW -alpha remove output/Figure_1.png output/Figure_1.tiff"
)
os.system("mogrify -alpha off output/Figure_1.tiff")

# ## Figure 2

article_distance_path = Path("../biorxiv/article_distances/output/figures")

# +
panel_one = sg.fromfile(article_distance_path / "biorxiv_article_distance.svg")

panel_one_size = (
    np.round(float(panel_one.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_one.root.attrib["height"][:-2]) * 1.33, 0),
)

scale_x = 1
scale_y = 1

print(f"original: {panel_one_size}")
print(f"scaled:{(panel_one_size[0]*scale_x, panel_one_size[1]*scale_y)}")

panel_one = panel_one.getroot()
panel_one.scale(x=scale_x, y=scale_y)
panel_one.moveto(20, 20)

# +
panel_two = sg.fromfile(article_distance_path / "distance_bin_accuracy.svg")

panel_two_size = (
    np.round(float(panel_two.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_two.root.attrib["height"][:-2]) * 1.33, 0),
)

scale_x = 1
scale_y = 1

print(f"original: {panel_two_size}")
print(f"scaled: {(panel_two_size[0]*scale_x, panel_two_size[1]*scale_y)}")

panel_two = panel_two.getroot()
panel_two.scale(x=scale_x, y=scale_y)
panel_two.moveto(580, 20)

# +
panel_three = sg.fromfile(
    article_distance_path / "publication_rate_reviewer_request.svg"
)

panel_three_size = (
    np.round(float(panel_three.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_three.root.attrib["height"][:-2]) * 1.33, 0),
)

scale_x = 1.3
scale_y = 1.3

print(f"original: {panel_three_size}")
print(f"scaled: {(panel_three_size[0]*scale_x, panel_three_size[1]*scale_y)}")

panel_three = panel_three.getroot()
panel_three.scale(x=scale_x, y=scale_y)
panel_three.moveto(40, 420)
# -

panel_one_label = sg.TextElement(10, 30, "A", size=30, weight="bold")
panel_two_label = sg.TextElement(560, 30, "B", size=30, weight="bold")
panel_three_label = sg.TextElement(10, 420, "C", size=30, weight="bold")

figure_two = sg.SVGFigure(Unit(1247), Unit(942))
figure_two.append(
    [
        etree.Element("rect", {"width": "100%", "height": "100%", "fill": "white"}),
        panel_one,
        panel_two,
        panel_three,
        panel_one_label,
        panel_two_label,
        panel_three_label,
    ]
)
display(SVG(figure_two.to_str()))

# save generated SVG files
figure_two.save("output/Figure_2.svg")
svg2png(bytestring=figure_two.to_str(), write_to="output/Figure_2.png", dpi=600)

os.system(
    "convert -compress LZW -alpha remove output/Figure_2.png output/Figure_2.tiff"
)
os.system("mogrify -alpha off output/Figure_2.tiff")

# ## Figure 3

publication_delay_path = Path("../biorxiv/publication_delay_experiment/output")
time_to_publication_path = Path("../biorxiv/time_to_publication/output")

# +
panel_one = sg.fromfile(time_to_publication_path / "preprint_category_halflife.svg")

panel_one_size = (
    np.round(float(panel_one.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_one.root.attrib["height"][:-2]) * 1.33, 0),
)

scale_x = 1.3
scale_y = 1.3

print(f"original: {panel_one_size}")
print(f"scaled:{(panel_one_size[0]*scale_x, panel_one_size[1]*scale_y)}")

panel_one = panel_one.getroot()
panel_one.scale(x=scale_x, y=scale_y)
panel_one.moveto(20, 20)

# +
panel_two = sg.fromfile(
    publication_delay_path / "version_count_vs_publication_time_violin.svg"
)

panel_two_size = (
    np.round(float(panel_two.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_two.root.attrib["height"][:-2]) * 1.33, 0),
)

scale_x = 1.3
scale_y = 1.3

print(f"original: {panel_two_size}")
print(f"scaled: {(panel_two_size[0]*scale_x, panel_two_size[1]*scale_y)}")

panel_two = panel_two.getroot()
panel_two.scale(x=scale_x, y=scale_y)
panel_two.moveto(900, 20)

# +
panel_three = sg.fromfile(
    publication_delay_path / "article_distance_vs_publication_time_hex.svg"
)

panel_three_size = (
    np.round(float(panel_three.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_three.root.attrib["height"][:-2]) * 1.33, 0),
)

scale_x = 1.3
scale_y = 1.3

print(f"original: {panel_three_size}")
print(f"scaled: {(panel_three_size[0]*scale_x, panel_three_size[1]*scale_y)}")

panel_three = panel_three.getroot()
panel_three.scale(x=scale_x, y=scale_y)
panel_three.moveto(100, 500)
# -

panel_one_label = sg.TextElement(10, 30, "A", size=30, weight="bold")
panel_two_label = sg.TextElement(900, 30, "B", size=30, weight="bold")
panel_three_label = sg.TextElement(20, 480, "C", size=30, weight="bold")

figure_three = sg.SVGFigure(Unit(1610), Unit(952))
figure_three.append(
    [
        etree.Element("rect", {"width": "100%", "height": "100%", "fill": "white"}),
        panel_one,
        panel_two,
        panel_three,
        panel_one_label,
        panel_two_label,
        panel_three_label,
    ]
)
display(SVG(figure_three.to_str()))

# save generated SVG files
figure_three.save("output/Figure_3.svg")
svg2png(bytestring=figure_three.to_str(), write_to="output/Figure_3.png", dpi=300)

os.system(
    "convert -compress LZW -alpha remove output/Figure_3.png output/Figure_3.tiff"
)
os.system("mogrify -alpha off output/Figure_3.tiff")

# ## Figure Five

polka_subset_path = Path("../biorxiv/polka_subset_experiment/output/figures")

# +
panel_one = sg.fromfile(polka_subset_path / "preprint_published_frequency_odds.svg")

panel_one_size = (
    np.round(float(panel_one.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_one.root.attrib["height"][:-2]) * 1.33, 0),
)

scale_x = 1.4
scale_y = 1.4

print(f"original: {panel_one_size}")
print(f"scaled:{(panel_one_size[0]*scale_x, panel_one_size[1]*scale_y)}")

panel_one = panel_one.getroot()
panel_one.scale(x=scale_x, y=scale_y)
panel_one.moveto(30, 30)

# +
panel_two = sg.fromfile(polka_subset_path / "preprint_published_frequency_bar.svg")

panel_two_size = (
    np.round(float(panel_two.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_two.root.attrib["height"][:-2]) * 1.33, 0),
)

scale_x = 1.4
scale_y = 1.4

print(f"original: {panel_two_size}")
print(f"scaled: {(panel_two_size[0]*scale_x, panel_two_size[1]*scale_y)}")

panel_two = panel_two.getroot()
panel_two.scale(x=scale_x, y=scale_y)
panel_two.moveto(1200, 10)

# +
panel_three = sg.fromfile(polka_subset_path / "saucie_plot.svg")

panel_three_size = (
    np.round(float(panel_three.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_three.root.attrib["height"][:-2]) * 1.33, 0),
)

scale_x = 1.2
scale_y = 1.2

print(f"original: {panel_three_size}")
print(f"scaled: {(panel_three_size[0]*scale_x, panel_three_size[1]*scale_y)}")

panel_three = panel_three.getroot()
panel_three.scale(x=scale_x, y=scale_y)
panel_three.moveto(50, 1000)

# +
panel_four = sg.fromfile(
    polka_subset_path / "version_count_vs_publication_time_violin.svg"
)

panel_four_size = (
    np.round(float(panel_four.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_four.root.attrib["height"][:-2]) * 1.33, 0),
)

scale_x = 1.4
scale_y = 1.4

print(f"original: {panel_four_size}")
print(f"scaled: {(panel_four_size[0]*scale_x, panel_three_size[1]*scale_y)}")

panel_four = panel_four.getroot()
panel_four.scale(x=scale_x, y=scale_y)
panel_four.moveto(1200, 900)

# +
panel_five = sg.fromfile(
    polka_subset_path / "article_distance_vs_publication_time_hex.svg"
)

panel_five_size = (
    np.round(float(panel_five.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_five.root.attrib["height"][:-2]) * 1.33, 0),
)

scale_x = 1.4
scale_y = 1.4

print(f"original: {panel_five_size}")
print(f"scaled: {(panel_five_size[0]*scale_x, panel_three_size[1]*scale_y)}")

panel_five = panel_five.getroot()
panel_five.scale(x=scale_x, y=scale_y)
panel_five.moveto(50, 1800)
# -

panel_one_label = sg.TextElement(10, 30, "A", size=30, weight="bold")
panel_two_label = sg.TextElement(1200, 30, "B", size=30, weight="bold")
panel_three_label = sg.TextElement(10, 900, "C", size=30, weight="bold")
panel_four_label = sg.TextElement(1200, 900, "D", size=30, weight="bold")
panel_five_label = sg.TextElement(10, 1850, "E", size=30, weight="bold")

figure_five = sg.SVGFigure(
    Unit(2250),
    Unit(2625),
)
figure_five.append(
    [
        etree.Element("rect", {"width": "100%", "height": "100%", "fill": "white"}),
        panel_one,
        panel_two,
        panel_three,
        panel_four,
        panel_five,
        panel_one_label,
        panel_two_label,
        panel_three_label,
        panel_four_label,
        panel_five_label,
    ]
)
display(SVG(figure_five.to_str()))

# save generated SVG files
figure_five.save("output/Figure_5.svg")
svg2png(bytestring=figure_five.to_str(), write_to="output/Figure_5.png", dpi=600)

os.system(
    "convert -compress LZW -alpha remove output/Figure_5.png output/Figure_5.tiff"
)
os.system("mogrify -alpha off output/Figure_5.tiff")

# # Supplemental Figures

# ## Figure S1

word_association_path = Path(
    "../biorxiv/pca_association_experiment/output/word_pca_similarity/figure_pieces"
)
pca_association_path = Path(
    "../biorxiv/pca_association_experiment/output/pca_plots/svg_files"
)

# +
panel_one = sg.fromfile(
    pca_association_path / "scatterplot_files/pca01_v_pca02_reversed.svg"
)

panel_one_size = (
    np.round(float(panel_one.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_one.root.attrib["height"][:-2]) * 1.33, 0),
)

scale_x = 1.2
scale_y = 1.2

print(f"original: {panel_one_size}")
print(f"scaled:{(panel_one_size[0]*scale_x, panel_one_size[1]*scale_y)}")

panel_one = panel_one.getroot()
panel_one.scale(x=scale_x, y=scale_y)
panel_one.moveto(20, 20)

# +
panel_two = load_clouds(
    str(word_association_path / "pca_01_cossim_word_cloud.png"), figure_size=(10, 7)
)

panel_two_size = (
    float(panel_two.root.attrib["width"]),
    float(panel_two.root.attrib["height"]),
)

scale_x = 1.4
scale_y = 1.4

print(f"original: {panel_two_size}")
print(f"scaled:{(panel_two_size[0]*scale_x, panel_two_size[1]*scale_y)}")

panel_two = panel_two.getroot()
panel_two.scale(x=scale_x, y=scale_y)
panel_two.moveto(1300, 100)

# +
panel_three = load_clouds(
    str(word_association_path / "pca_02_cossim_word_cloud.png"), figure_size=(10, 7)
)

panel_three_size = (
    float(panel_three.root.attrib["width"]),
    float(panel_three.root.attrib["height"]),
)

scale_x = 1.4
scale_y = 1.4

print(f"original: {panel_three_size}")
print(f"scaled:{(panel_three_size[0]*scale_x, panel_three_size[1]*scale_y)}")

panel_three = panel_three.getroot()
panel_three.scale(x=scale_x, y=scale_y)
panel_three.moveto(200, 830)

# +
panel_four = sg.fromfile(
    f"{str(pca_association_path)}/category_box_plot/category_box_plot_pc1.svg"
)


panel_four_size = (
    np.round(float(panel_four.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_four.root.attrib["height"][:-2]) * 1.33, 0),
)

scale_x = 1.4
scale_y = 1.4

print(f"original: {panel_four_size}")
print(f"scaled:{(panel_four_size[0]*scale_x, panel_four_size[1]*scale_y)}")

panel_four = panel_four.getroot()
panel_four.scale(x=scale_x, y=scale_y)
panel_four.moveto(1000, 800)

# +
panel_five = sg.fromfile(
    pca_association_path / "category_box_plot/category_box_plot_pc2.svg"
)


panel_five_size = (
    np.round(float(panel_five.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_five.root.attrib["height"][:-2]) * 1.33, 0),
)

scale_x = 1.4
scale_y = 1.4

print(f"original: {panel_five_size}")
print(f"scaled:{(panel_five_size[0]*scale_x, panel_five_size[1]*scale_y)}")

panel_five = panel_five.getroot()
panel_five.scale(x=scale_x, y=scale_y)
panel_five.moveto(20, 1600)
# -

panel_one_label = sg.TextElement(20, 50, "A", size=50, weight="bold")
panel_one_image_label = sg.TextElement(1600, 670, "PC 1", size=40, weight="bold")
panel_two_label = sg.TextElement(1200, 50, "B", size=50, weight="bold")
panel_two_image_label = sg.TextElement(500, 1410, "PC 2", size=40, weight="bold")
panel_three_label = sg.TextElement(20, 800, "C", size=50, weight="bold")
panel_four_label = sg.TextElement(1000, 800, "D", size=50, weight="bold")
panel_five_label = sg.TextElement(20, 1600, "E", size=50, weight="bold")

# +
figure_S1 = sg.SVGFigure(Unit(2220), Unit(2348))

figure_S1.append(
    [
        etree.Element("rect", {"width": "100%", "height": "100%", "fill": "white"}),
        panel_one,
        panel_two,
        panel_three,
        panel_four,
        panel_five,
        panel_one_label,
        panel_one_image_label,
        panel_two_label,
        panel_two_image_label,
        panel_three_label,
        panel_four_label,
        panel_five_label,
    ]
)
display(SVG(figure_S1.to_str()))
# -

# save generated SVG files
figure_S1.save("output/Figure_S1.svg")
svg2png(bytestring=figure_S1.to_str(), write_to="output/Figure_S1.png", dpi=600)

os.system(
    "convert -compress LZW -alpha remove output/Figure_S1.png output/Figure_S1.tiff"
)
os.system("mogrify -alpha off output/Figure_S1.tiff")

# ## Figure S2

biorxiv_section_path = Path("../biorxiv/exploratory_data_analysis/output")

# +
panel_one = sg.fromfile(biorxiv_section_path / "figures/preprint_category.svg")

panel_one_size = (
    np.round(float(panel_one.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_one.root.attrib["height"][:-2]) * 1.33, 0),
)

scale_x = 1
scale_y = 1

print(f"original: {panel_one_size}")
print(f"scaled:{(panel_one_size[0]*scale_x, panel_one_size[1]*scale_y)}")

panel_one = panel_one.getroot()
panel_one.scale(x=scale_x, y=scale_y)
panel_one.moveto(20, 20)

# +
figure_S2 = sg.SVGFigure(Unit(646), Unit(335))

figure_S2.append(
    [
        etree.Element("rect", {"width": "100%", "height": "100%", "fill": "white"}),
        panel_one,
    ]
)

display(SVG(figure_S2.to_str()))
# -

# save generated SVG files
figure_S2.save("output/Figure_S2.svg")
svg2png(bytestring=figure_S2.to_str(), write_to="output/Figure_S2.png", dpi=600)

os.system(
    "convert -compress LZW -alpha remove output/Figure_S2.png output/Figure_S2.tiff"
)
os.system("mogrify -alpha off output/Figure_S2.tiff")

# ## Figure S3

corpora_comparison_path = Path("../biorxiv/corpora_comparison/output/")

# +
panel_one = sg.fromfile(
    corpora_comparison_path
    / "svg_files/biorxiv_pmc_frequency_odds_special_char_removed.svg"
)

panel_one_size = (
    np.round(float(panel_one.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_one.root.attrib["height"][:-2]) * 1.33, 0),
)

scale_x = 1
scale_y = 1

print(f"original: {panel_one_size}")
print(f"scaled:{(panel_one_size[0]*scale_x, panel_one_size[1]*scale_y)}")

panel_one = panel_one.getroot()
panel_one.scale(x=scale_x, y=scale_y)
panel_one.moveto(20, 50)

# +
panel_two = sg.fromfile(
    corpora_comparison_path
    / "svg_files/biorxiv_pmc_frequency_bar_special_char_removed.svg"
)

panel_two_size = (
    np.round(float(panel_two.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_two.root.attrib["height"][:-2]) * 1.33, 0),
)

scale_x = 1
scale_y = 1

print(f"original: {panel_two_size}")
print(f"scaled:{(panel_two_size[0]*scale_x, panel_two_size[1]*scale_y)}")

panel_two = panel_two.getroot()
panel_two.scale(x=scale_x, y=scale_y)
panel_two.moveto(850, 30)
# -

panel_one_label = sg.TextElement(20, 50, "A", size=20, weight="bold")
panel_two_label = sg.TextElement(830, 50, "B", size=20, weight="bold")

# +
figure_S3 = sg.SVGFigure(Unit(1563), Unit(576))

figure_S3.append(
    [
        etree.Element("rect", {"width": "100%", "height": "100%", "fill": "white"}),
        panel_one,
        panel_two,
        panel_one_label,
        panel_two_label,
    ]
)

display(SVG(figure_S3.to_str()))
# -

# save generated SVG files
figure_S3.save("output/Figure_S3.svg")
svg2png(bytestring=figure_S3.to_str(), write_to="output/Figure_S3.png", dpi=600)

os.system(
    "convert -compress LZW -alpha remove output/Figure_S3.png output/Figure_S3.tiff"
)
os.system("mogrify -alpha off output/Figure_S3.tiff")

# ## Figure S4

corpora_comparison_path = Path("../biorxiv/corpora_comparison/output/")

# +
panel_one = sg.fromfile(
    corpora_comparison_path
    / "svg_files/preprint_published_frequency_odds_special_char_removed.svg"
)

panel_one_size = (
    np.round(float(panel_one.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_one.root.attrib["height"][:-2]) * 1.33, 0),
)

scale_x = 1
scale_y = 1

print(f"original: {panel_one_size}")
print(f"scaled:{(panel_one_size[0]*scale_x, panel_one_size[1]*scale_y)}")

panel_one = panel_one.getroot()
panel_one.scale(x=scale_x, y=scale_y)
panel_one.moveto(20, 50)

# +
panel_two = sg.fromfile(
    corpora_comparison_path
    / "svg_files/preprint_published_frequency_bar_special_char_removed.svg"
)

panel_two_size = (
    np.round(float(panel_two.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_two.root.attrib["height"][:-2]) * 1.33, 0),
)

scale_x = 1
scale_y = 1

print(f"original: {panel_two_size}")
print(f"scaled:{(panel_two_size[0]*scale_x, panel_two_size[1]*scale_y)}")

panel_two = panel_two.getroot()
panel_two.scale(x=scale_x, y=scale_y)
panel_two.moveto(850, 30)
# -

panel_one_label = sg.TextElement(20, 50, "A", size=20, weight="bold")
panel_two_label = sg.TextElement(830, 50, "B", size=20, weight="bold")

# +
figure_S4 = sg.SVGFigure(Unit(1563), Unit(576))

figure_S4.append(
    [
        etree.Element("rect", {"width": "100%", "height": "100%", "fill": "white"}),
        panel_one,
        panel_two,
        panel_one_label,
        panel_two_label,
    ]
)

display(SVG(figure_S4.to_str()))
# -

# save generated SVG files
figure_S4.save("output/Figure_S4.svg")
svg2png(bytestring=figure_S4.to_str(), write_to="output/Figure_S4.png", dpi=600)

os.system(
    "convert -compress LZW -alpha remove output/Figure_S4.png output/Figure_S4.tiff"
)
os.system("mogrify -alpha off output/Figure_S4.tiff")

# ## Figure S5

knn_results_path = Path("../pmc/journal_recommendation/output")

# +
panel_one = sg.fromfile(knn_results_path / "figures/knn_result.svg")

panel_one_size = (
    np.round(float(panel_one.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_one.root.attrib["height"][:-2]) * 1.33, 0),
)

scale_x = 1
scale_y = 1

print(f"original: {panel_one_size}")
print(f"scaled:{(panel_one_size[0]*scale_x, panel_one_size[1]*scale_y)}")

panel_one = panel_one.getroot()
panel_one.scale(x=scale_x, y=scale_y)
panel_one.moveto(20, 50)

# +
figure_S5 = sg.SVGFigure(Unit(640), Unit(458))

figure_S5.append(
    [
        etree.Element("rect", {"width": "100%", "height": "100%", "fill": "white"}),
        panel_one,
    ]
)

display(SVG(figure_S5.to_str()))
# -

# save generated SVG files
figure_S5.save("output/Figure_S5.svg")
svg2png(bytestring=figure_S5.to_str(), write_to="output/Figure_S5.png", dpi=600)

os.system(
    "convert -compress LZW -alpha remove output/Figure_S5.png output/Figure_S5.tiff"
)
os.system("mogrify -alpha off output/Figure_S5.tiff")

# ## Figure S6

polka_subset_path = Path("../biorxiv/polka_subset_experiment/output/figures")

# +
panel_one = sg.fromfile(
    polka_subset_path / "version_count_vs_publication_time_violin_filtered.svg"
)

panel_one_size = (
    np.round(float(panel_one.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_one.root.attrib["height"][:-2]) * 1.33, 0),
)

scale_x = 1
scale_y = 1

print(f"original: {panel_one_size}")
print(f"scaled:{(panel_one_size[0]*scale_x, panel_one_size[1]*scale_y)}")

panel_one = panel_one.getroot()
panel_one.scale(x=scale_x, y=scale_y)
panel_one.moveto(0, 0)

# +
panel_two = sg.fromfile(
    polka_subset_path / "article_distance_vs_publication_time_hex_filtered.svg"
)

panel_two_size = (
    np.round(float(panel_two.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_two.root.attrib["height"][:-2]) * 1.33, 0),
)

scale_x = 1
scale_y = 1

print(f"original: {panel_two_size}")
print(f"scaled: {(panel_two_size[0]*scale_x, panel_two_size[1]*scale_y)}")

panel_two = panel_two.getroot()
panel_two.scale(x=scale_x, y=scale_y)
panel_two.moveto(750, 0)
# -

panel_one_label = sg.TextElement(10, 30, "A", size=30, weight="bold")
panel_two_label = sg.TextElement(750, 30, "B", size=30, weight="bold")

figure_S6 = sg.SVGFigure(Unit(1469), Unit(567))
figure_S6.append(
    [
        etree.Element("rect", {"width": "100%", "height": "100%", "fill": "white"}),
        panel_one,
        panel_two,
        panel_one_label,
        panel_two_label,
    ]
)
display(SVG(figure_S6.to_str()))

# save generated SVG files
figure_S6.save("output/Figure_S6.svg")
svg2png(bytestring=figure_S6.to_str(), write_to="output/Figure_S6.png", dpi=600)

os.system(
    "convert -compress LZW -alpha remove output/Figure_S6.png output/Figure_S6.tiff"
)
os.system("mogrify -alpha off output/Figure_S6.tiff")
