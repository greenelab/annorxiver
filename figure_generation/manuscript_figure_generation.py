#!/usr/bin/env python
# coding: utf-8

# # Figure Generation for Manuscript

# In[1]:


from pathlib import Path
from IPython.display import Image, display, SVG

from cairosvg import svg2png
from lxml import etree
import numpy as np
import svgutils.transform as sg


# ## Figure 1

# In[2]:


corpora_comparison_path = Path("../biorxiv/corpora_comparison/output/")


# In[3]:


panel_one = (
    sg.fromfile(corpora_comparison_path/"svg_files/corpora_kl_divergence.svg")
)

# Convert pt units to pixel units
# Vince's tutorial FTW
panel_one_size = (
    np.round(float(panel_one.root.attrib['width'][:-2])*1.33, 0), 
    np.round(float(panel_one.root.attrib['height'][:-2])*1.33, 0)
)

scale_x = 1.5
scale_y = 1.5

print(f"original: {panel_one_size}")
print(f"scaled:{(panel_one_size[0]*scale_x,panel_one_size[1]*scale_y)}")

panel_one = panel_one.getroot()
panel_one.scale_xy(x=scale_x, y=scale_y)
panel_one.moveto(100,30)


# In[4]:


panel_two = (
    sg.fromfile(corpora_comparison_path/"svg_files/biorxiv_pmc_frequency_odds.svg")
)

panel_two_size = (
    np.round(float(panel_two.root.attrib['width'][:-2])*1.33, 0), 
    np.round(float(panel_two.root.attrib['height'][:-2])*1.33, 0),
)
scale_x = 1.2
scale_y = 1.2

print(f"original: {panel_two_size}")
print(f"scaled:{(panel_two_size[0]*scale_x, panel_two_size[1]*scale_y)}")

panel_two = panel_two.getroot()
panel_two.scale_xy(x=scale_x, y=scale_y)
panel_two.moveto(30, 600)


# In[5]:


panel_three = (
    sg.fromfile(corpora_comparison_path/"svg_files/biorxiv_pmc_frequency_bar.svg")
)

panel_three_size = (
    np.round(float(panel_three.root.attrib['width'][:-2])*1.33, 0), 
    np.round(float(panel_three.root.attrib['height'][:-2])*1.33, 0),
)
scale_x = 1.2
scale_y = 1.2

print(f"original: {panel_three_size}")
print(f"scaled:{(panel_three_size[0]*scale_x, panel_three_size[1]*scale_y)}")

panel_three = panel_three.getroot()
panel_three.scale_xy(x=scale_x, y=scale_y)
panel_three.moveto(630, 580)


# In[6]:


panel_four = (
    sg.fromfile(corpora_comparison_path/"svg_files/preprint_published_frequency_odds.svg")
)

panel_four_size = (
    np.round(float(panel_four.root.attrib['width'][:-2])*1.33, 0), 
    np.round(float(panel_four.root.attrib['height'][:-2])*1.33, 0),
)
scale_x = 1.15
scale_y = 1.15

print(f"original: {panel_four_size}")
print(f"scaled:{(panel_four_size[0]*scale_x, panel_four_size[1]*scale_y)}")

panel_four = panel_four.getroot()
panel_four.scale_xy(x=scale_x, y=scale_y)
panel_four.moveto(30, 1020, scale=1)


# In[7]:


panel_five = (
    sg.fromfile(corpora_comparison_path/"svg_files/preprint_published_frequency_bar.svg")

)

panel_five_size = (
    np.round(float(panel_five.root.attrib['width'][:-2])*1.33, 0), 
    np.round(float(panel_five.root.attrib['height'][:-2])*1.33, 0),
)
scale_x = 1.15
scale_y = 1.15

print(f"original: {panel_five_size}")
print(f"scaled:{(panel_five_size[0]*scale_x, panel_five_size[1]*scale_y)}")

panel_five = panel_five.getroot()
panel_five.scale_xy(x=scale_x, y=scale_y)
panel_five.moveto(620, 1000, scale=1)


# In[8]:


panel_one_label = sg.TextElement(10, 20, "A", size=22, weight="bold")
panel_two_label = sg.TextElement(10, 600, "B", size=22, weight="bold")
panel_three_label = sg.TextElement(610, 600, "C", size=22, weight="bold")
panel_four_label = sg.TextElement(30, 1010, "D", size=22, weight="bold")
panel_five_label = sg.TextElement(620, 1010, "E", size=22, weight="bold")


# In[9]:


figure_one = sg.SVGFigure("1600", "1400")
figure_one.append([
    etree.Element("rect", {"width":"100%", "height":"100%", "fill":"white"}),
    panel_one, 
    panel_two, 
    panel_three,
    panel_four,
    panel_five,
    panel_one_label,
    panel_two_label,
    panel_three_label,
    panel_four_label,
    panel_five_label
])
display(SVG(figure_one.to_str()))


# In[10]:


# save generated SVG files
figure_one.save("output/figure_one_panels.svg")


# ## Figure 2

# In[11]:


word_association_path = Path(
    "../biorxiv/pca_association_experiment/output/word_pca_similarity/svg_files"
)
pca_association_path = Path(
    "../biorxiv/pca_association_experiment/output/pca_plots/svg_files"
)


# In[12]:


panel_one = (
    sg.fromfile(pca_association_path/"scatterplot_files/pca01_v_pca02_reversed.svg")
)

panel_one_size = (
    np.round(float(panel_one.root.attrib['width'][:-2])*1.33, 0), 
    np.round(float(panel_one.root.attrib['height'][:-2])*1.33, 0)
)

scale_x = 1
scale_y = 1

print(f"original: {panel_one_size}")
print(f"scaled:{(panel_one_size[0]*scale_x, panel_one_size[1]*scale_y)}")

panel_one = panel_one.getroot()
panel_one.scale_xy(x=scale_x, y=scale_y)
panel_one.moveto(20,20)


# In[13]:


panel_two = (
    sg.fromfile(word_association_path/"pca_02_cossim_word_cloud.svg")
)

panel_two_size = (
    float(panel_two.root.attrib['width']),
    float(panel_two.root.attrib['height'])
)

scale_x = 0.45
scale_y = 0.45

print(f"original: {panel_two_size}")
print(f"scaled:{(panel_two_size[0]*scale_x, panel_two_size[1]*scale_y)}")

panel_two = panel_two.getroot()
panel_two.scale_xy(x=scale_x, y=scale_y)
panel_two.moveto(620,20)


# In[14]:


panel_three = (
    sg.fromfile(word_association_path/"pca_01_cossim_word_cloud.svg")
)

panel_three_size = (
    float(panel_three.root.attrib['width']),
    float(panel_three.root.attrib['height'])
)

scale_x = 0.45
scale_y = 0.45

print(f"original: {panel_three_size}")
print(f"scaled:{(panel_three_size[0]*scale_x, panel_three_size[1]*scale_y)}")

panel_three = panel_three.getroot()
panel_three.scale_xy(x=scale_x, y=scale_y)
panel_three.moveto(30,430)


# In[15]:


panel_four = (
    sg.fromfile(f"{str(pca_association_path)}/category_box_plot/category_box_plot_pc2.svg")
)


panel_four_size = (
    np.round(float(panel_four.root.attrib['width'][:-2])*1.33, 0), 
    np.round(float(panel_four.root.attrib['height'][:-2])*1.33, 0)
)

scale_x = 1.3
scale_y = 1.3

print(f"original: {panel_four_size}")
print(f"scaled:{(panel_four_size[0]*scale_x, panel_four_size[1]*scale_y)}")

panel_four = panel_four.getroot()
panel_four.scale_xy(x=scale_x, y=scale_y)
panel_four.moveto(615,430)


# In[16]:


panel_five = (
    sg.fromfile(pca_association_path/"category_box_plot/category_box_plot_pc1.svg")
)


panel_five_size = (
    np.round(float(panel_five.root.attrib['width'][:-2])*1.33, 0), 
    np.round(float(panel_five.root.attrib['height'][:-2])*1.33, 0)
)

scale_x = 1.3
scale_y = 1.3

print(f"original: {panel_five_size}")
print(f"scaled:{(panel_five_size[0]*scale_x, panel_five_size[1]*scale_y)}")

panel_five = panel_five.getroot()
panel_five.scale_xy(x=scale_x, y=scale_y)
panel_five.moveto(15,860)


# In[17]:


panel_one_label = sg.TextElement(10, 20, "A", size=22, weight="bold")
panel_one_image_label = sg.TextElement(210, 800, "PCA 1", size=22, weight="bold")
panel_two_label = sg.TextElement(600, 20, "B", size=22, weight="bold")
panel_two_image_label = sg.TextElement(820, 385, "PCA 2", size=22, weight="bold")
panel_three_label = sg.TextElement(10, 420, "C", size=22, weight="bold")
panel_four_label = sg.TextElement(600, 420, "D", size=22, weight="bold")
panel_five_label = sg.TextElement(10, 850, "E", size=22, weight="bold")


# In[18]:


figure_two = sg.SVGFigure("1600", "1400")
figure_two.append([
    etree.Element("rect", {"width":"100%", "height":"100%", "fill":"white"}),
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
])
display(SVG(figure_two.to_str()))


# In[19]:


# save generated SVG files
figure_two.save("output/figure_two_panels.svg")


# ## Figure 3

# In[20]:


article_distance_path = Path(
    "../biorxiv/article_distances/output/figures"
)


# In[21]:


panel_one = (
    sg.fromfile(article_distance_path/"biorxiv_article_distance.svg")
)

panel_one_size = (
    np.round(float(panel_one.root.attrib['width'][:-2])*1.33, 0),
    np.round(float(panel_one.root.attrib['height'][:-2])*1.33, 0)
)

scale_x = 1
scale_y = 1

print(f"original: {panel_one_size}")
print(f"scaled:{(panel_one_size[0]*scale_x, panel_one_size[1]*scale_y)}")

panel_one = panel_one.getroot()
panel_one.scale_xy(x=scale_x, y=scale_y)
panel_one.moveto(20,20)


# In[22]:


panel_two = (
    sg.fromfile(article_distance_path/"distance_bin_accuracy.svg")
)

panel_two_size = (
    np.round(float(panel_two.root.attrib['width'][:-2])*1.33, 0),
    np.round(float(panel_two.root.attrib['height'][:-2])*1.33, 0)
)

scale_x = 1
scale_y = 1

print(f"original: {panel_two_size}")
print(f"scaled: {(panel_two_size[0]*scale_x, panel_two_size[1]*scale_y)}")

panel_two = panel_two.getroot()
panel_two.scale_xy(x=scale_x, y=scale_y)
panel_two.moveto(580,20)


# In[23]:


panel_three = (
    sg.fromfile(article_distance_path/"publication_rate.svg")
)

panel_three_size = (
    np.round(float(panel_three.root.attrib['width'][:-2])*1.33, 0),
    np.round(float(panel_three.root.attrib['height'][:-2])*1.33, 0)
)

scale_x = 1
scale_y = 1

print(f"original: {panel_three_size}")
print(f"scaled: {(panel_three_size[0]*scale_x, panel_three_size[1]*scale_y)}")

panel_three = panel_three.getroot()
panel_three.scale_xy(x=scale_x, y=scale_y)
panel_three.moveto(20,420)


# In[24]:


panel_one_label = sg.TextElement(10, 20, "A", size=22, weight="bold")
panel_two_label = sg.TextElement(560, 20, "B", size=22, weight="bold")
panel_three_label = sg.TextElement(10, 420, "C", size=22, weight="bold")


# In[25]:


figure_three = sg.SVGFigure("1400", "800")
figure_three.append([
    etree.Element("rect", {"width":"100%", "height":"100%", "fill":"white"}),
    panel_one,
    panel_two,
    panel_three,
    panel_one_label,
    panel_two_label,
    panel_three_label
])
display(SVG(figure_three.to_str()))


# In[26]:


# save generated SVG files
figure_three.save("output/figure_three_panels.svg")


# ## Figure 4

# In[27]:


publication_delay_path = Path(
    "../biorxiv/publication_delay_experiment/output"
)
time_to_publication_path = Path(
    "../biorxiv/time_to_publication/output"
)


# In[28]:


panel_one = (
    sg.fromfile(publication_delay_path/"article_distance_vs_publication_time.svg")
)

panel_one_size = (
    np.round(float(panel_one.root.attrib['width'][:-2])*1.33, 0),
    np.round(float(panel_one.root.attrib['height'][:-2])*1.33, 0)
)

scale_x = 1
scale_y = 1

print(f"original: {panel_one_size}")
print(f"scaled:{(panel_one_size[0]*scale_x, panel_one_size[1]*scale_y)}")

panel_one = panel_one.getroot()
panel_one.scale_xy(x=scale_x, y=scale_y)
panel_one.moveto(20,20)


# In[29]:


panel_two = (
    sg.fromfile(publication_delay_path/"version_count_vs_publication_time.svg")
)

panel_two_size = (
    np.round(float(panel_two.root.attrib['width'][:-2])*1.33, 0),
    np.round(float(panel_two.root.attrib['height'][:-2])*1.33, 0)
)

scale_x = 1
scale_y = 1

print(f"original: {panel_two_size}")
print(f"scaled: {(panel_two_size[0]*scale_x, panel_two_size[1]*scale_y)}")

panel_two = panel_two.getroot()
panel_two.scale_xy(x=scale_x, y=scale_y)
panel_two.moveto(580,20)


# In[30]:


panel_three = (
    sg.fromfile(time_to_publication_path/"preprint_category_halflife.svg")
)

panel_three_size = (
    np.round(float(panel_three.root.attrib['width'][:-2])*1.33, 0),
    np.round(float(panel_three.root.attrib['height'][:-2])*1.33, 0)
)

scale_x = 1
scale_y = 1

print(f"original: {panel_three_size}")
print(f"scaled: {(panel_three_size[0]*scale_x, panel_three_size[1]*scale_y)}")

panel_three = panel_three.getroot()
panel_three.scale_xy(x=scale_x, y=scale_y)
panel_three.moveto(20,420)


# In[31]:


panel_one_label = sg.TextElement(10, 20, "A", size=22, weight="bold")
panel_two_label = sg.TextElement(560, 20, "B", size=22, weight="bold")
panel_three_label = sg.TextElement(10, 420, "C", size=22, weight="bold")


# In[32]:


figure_four = sg.SVGFigure("1400", "800")
figure_four.append([
    etree.Element("rect", {"width":"100%", "height":"100%", "fill":"white"}),
    panel_one,
    panel_two,
    panel_three,
    panel_one_label,
    panel_two_label,
    panel_three_label
])
display(SVG(figure_four.to_str()))


# In[33]:


# save generated SVG files
figure_four.save("output/figure_four_panels.svg")

