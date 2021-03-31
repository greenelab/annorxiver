library(ggplot2)

bin_num <- 50
g <- (
     ggplot(pmc_data_df, mapping=aes(x=dim1, y=dim2))
     + geom_bin2d(bins=bin_num, bin_width=85)
     + geom_point(data=subset_df, mapping=aes(x=dim1, y=dim2, color='red'))
     + theme(legend.position="left")
)
ggsave(file='output/figures/saucie_plot.png', plot=g, width=8, height=6)