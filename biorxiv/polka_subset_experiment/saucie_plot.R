library(ggplot2)

bin_width <- 0.85
g <- (
     ggplot(pmc_data_df, mapping=aes(x=dim1, y=dim2))
     + geom_bin2d(binwidth=bin_width)
     + scale_fill_gradient(low="#253494", high="#edf8b1")
     + geom_point(data=subset_df, mapping=aes(x=dim1, y=dim2, color='red'), show.legend = FALSE)
     + theme(legend.position="left", text=element_text(size=12))
)
ggsave(file='output/figures/saucie_plot.svg', plot=g, width=11, height=8, dpi=600)
ggsave(file='output/figures/saucie_plot.png', plot=g, width=11, height=8, dpi=600)