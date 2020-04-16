from cairosvg import svg2png
from IPython.display import Image, HTML, display
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from pathlib import Path
import plotnine as p9
import re
import svgutils.transform as sg

def display_clouds(pc_cloud_1, pc_cloud_2):
    return display(
        HTML(
            f"""
            <table>
                <tr>
                    <td>
                    <img src={pc_cloud_1}>
                    </td>
                    <td>
                    <img src={pc_cloud_2}>
                    </td>
                </tr>
            </table>
            """
        )
    )
    
def save_plot(image_path, output_path):
    img = mpimg.imread(image_path)
    fig = plt.figure(figsize=(8,6), dpi=300)
    plt.imshow(img)
    plt.axis("off")
    fig.savefig(output_path, format='svg')
    #Turn off the images
    plt.close()
    

def plot_scatter_clouds(
    scatter_plot_path, word_cloud_x_path, 
    word_cloud_y_path, final_figure_path="output/pca_plots/figures/final_figure.png"
):
    
    # Save the matplotlfigures
    scatter_plot = Path(scatter_plot_path)
    scatter_plot_output = f"output/pca_plots/svg_files/{scatter_plot.stem}.svg"
   
    word_cloud_x = Path(word_cloud_x_path)
    word_cloud_x_output = f"output/pca_plots/svg_files/{word_cloud_x.stem}.svg"
    
    word_cloud_y = Path(word_cloud_y_path)
    word_cloud_y_output = f"output/pca_plots/svg_files/{word_cloud_y.stem}.svg"
    
    save_plot(str(scatter_plot), scatter_plot_output)
    save_plot(str(word_cloud_x), word_cloud_x_output)
    save_plot(str(word_cloud_y), word_cloud_y_output)
    
    #create new SVG figure
    fig = sg.SVGFigure("1280", "768")

    # load matpotlib-generated figures
    fig1 = sg.fromfile(word_cloud_y_output)
    fig2 = sg.fromfile(scatter_plot_output)
    fig3 = sg.fromfile(word_cloud_x_output)

    # get the plot objects
    plot1 = fig1.getroot()
    plot1.moveto(-40, -20, scale=0.98)

    plot2 = fig2.getroot()
    plot2.moveto(420, -110, scale=1.4)
    
    plot3 = fig3.getroot()
    plot3.moveto(460, 370, scale=0.98)

    # append plots and labels to figure
    fig.append([plot2,plot1, plot3])
    
    text_A = sg.TextElement(10, 30, "A", size=20, weight="bold")
    text_B = sg.TextElement(510, 30, "B", size=20, weight="bold")
    text_C = sg.TextElement(510, 390, "C", size=20, weight="bold")
    
    fig.append([text_A, text_B, text_C])
    
    second_pc = int(
        re
        .search(r"pca_(\d+)", word_cloud_y.stem)
        .group(1)
    )
    
    first_pc = int(
        re
        .search(r"pca_(\d+)", word_cloud_x.stem)
        .group(1)
    )
    
    
    word_cloud_title_1 = sg.TextElement(
        200, 370, 
        f"pca{second_pc}", size=22, 
        weight="bold"
    )
    
    word_cloud_title_2 = sg.TextElement(
        713, 760, 
        f"pca{first_pc}", size=22, 
        weight="bold"
    )
    
    fig.append([word_cloud_title_1, word_cloud_title_2])

    # save generated SVG files
    svg2png(
        bytestring=fig.to_str(), 
        write_to=final_figure_path,
        dpi=500
    )
    
    return Image(final_figure_path)


def generate_scatter_plots(
    data,
    x="pca1", y="pca2", 
    nsample=200, random_state=100,
    selected_categories=['bioinformatics', 'neuroscience'],
    color_palette=['#a6cee3','#1f78b4'],
    save_file_path="output/pca_plots/scatterplot_files/pca01_v_pca02.png"
):
    g = (
        p9.ggplot(
            data
            .query(f"category in {selected_categories}")
            .groupby("category")
            .apply(lambda x: x.sample(nsample, random_state=random_state) if len(x) > nsample else x)
            .reset_index(drop=True)
        )
        + p9.aes(x=x, y=y, color="factor(category)")
        + p9.geom_point()
        + p9.scale_color_manual(
            {
                category:color
                for category, color in zip(selected_categories, color_palette)
            }
        )
        + p9.labs(
            title="PCA of BioRxiv (Word Dim: 300)",
            color="Article Category"
        )
        + p9.theme(
            figure_size=(4.59,3.44),
            dpi=300
        )
    )
    
    g.save(save_file_path)
    print(g)