
plotEcosystem=function(spata_obj, subgraph_nodes, subgraph_index, extend=0.2){
  library(ggforce)
  plot_df <- SPATA2::getCoordsDf(spata_obj)
  celltype_df = spata_obj@spatial[[1]]$cytospace
  
  ## isolate the barcodes of the defines subgraph:
  sub <- subgraph_nodes[[subgraph_index]]
  
  
  plot_df$graph=1
  plot_df[plot_df$barcodes %in% sub, ]$graph=0.2
  
  cells <- celltype_df %>% filter(barcodes %in% sub)
  cell_ID <- cells$UniqueCID
  
  ## Define ranges: 
  x_range <- cells %>% filter(barcodes %in% sub) %>% pull(x_sc) %>% range()
  y_range <- cells %>% filter(barcodes %in% sub) %>% pull(y_sc) %>% range()
  
  ## Modify ranges:
  x_range[1] <- x_range[1]-c(x_range[2]-x_range[1])*extend
  x_range[2] <- x_range[2]+c(x_range[2]-x_range[1])*extend
  y_range[1] <- y_range[1]-c(y_range[2]-y_range[1])*extend
  y_range[2] <- y_range[2]+c(y_range[2]-y_range[1])*extend
  
  plot_df_cells <-
    celltype_df %>%
    filter(x_sc>x_range[1] & x_sc<x_range[2]) %>%
    filter(y_sc>y_range[1] & y_sc<y_range[2]) %>%
    mutate(Graph=ifelse(barcodes %in% sub, T, F))
  
  rownames(plot_df_cells) <- plot_df_cells$UniqueCID
  plot_df_cells[plot_df_cells$Graph==F, ]$CellType="no"
  cc <- c(cc, no="#FFFFFF")
  
  p <- ggplot(plot_df_cells, aes(x_sc, y_sc, group = -1L)) +
    geom_voronoi_tile(aes(fill = CellType), max.radius = 10, colour = 'black', linewidth=0.1)+
    scale_fill_manual(values=cc)+
    theme_bw() +
    theme(panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_rect(colour = "black", size=0.5),
          axis.text.x = element_text(colour="black"),
          axis.text.y = element_text(colour="black"))+
    xlim(x_range)+
    ylim(y_range)+
    coord_fixed()
  #Seurat::NoLegend()
  
  return(p)
  
}
