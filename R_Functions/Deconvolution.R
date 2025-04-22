

#' Export scRNA-seq Data for Cytospace Mapping
#'
#' This function exports a Seurat single-cell RNA-seq object into two tab-delimited text files: one containing the expression matrix and another with cell-type labels, in a format compatible with Cytospace.
#'
#' @param scrna_seurat A Seurat object containing scRNA-seq data with associated cell type annotations.
#' @param dir_out A character string specifying the output directory. If empty (`''`), files will be written to the current working directory.
#' @param fout_prefix A character string to prepend to the output filenames.
#' @param rna_assay A character string indicating the name of the RNA assay to use from the Seurat object. Default is `"RNA"`.
#'
#' @return This function writes two files to disk:
#' \describe{
#'   \item{`<prefix>scRNA_data.txt`}{A tab-delimited file containing gene expression values. Rows are genes, columns are cells.}
#'   \item{`<prefix>cell_type_labels.txt`}{A tab-delimited file with cell IDs and their associated cell type annotations from the metadata field `annotation_level_4`.}
#' }
#'
#' @details
#' The function uses `GetAssayData` to extract the raw count matrix, and assumes the Seurat object contains a metadata column named `annotation_level_4` representing cell type labels. It writes the output files using `data.table::fwrite` for speed.
#'
#' If `dir_out` does not exist, it will be created. If `fout_prefix` is set, it will be added as a prefix to the filenames.
#'
#' @importFrom Seurat GetAssayData
#' @importFrom data.table fwrite
#'
#' @examples
#' \dontrun{
#' generate_cytospace_from_scRNA_seurat_object(
#'   scrna_seurat = my_seurat_object,
#'   dir_out = "cytospace_inputs",
#'   fout_prefix = "sample1_",
#'   rna_assay = "RNA"
#' )
#' }
#'
#' @export
generate_cytospace_from_scRNA_seurat_object <- function(scrna_seurat, dir_out = '', fout_prefix = '', rna_assay = 'RNA') {
  scrna_count <- as.data.frame(as.matrix(GetAssayData(object = scrna_seurat, slot = "counts", assay = rna_assay)))
  cell_names <- colnames(scrna_count)
  scrna_count <- cbind(rownames(scrna_count), scrna_count)
  colnames(scrna_count)[1] <- 'GENES'
  
  cell_type_labels <- data.frame(scrna_seurat$celltype)
  rownames(cell_type_labels) <- cell_names
  cell_type_labels <- cbind(rownames(cell_type_labels), cell_type_labels)
  colnames(cell_type_labels) <- c('Cell IDs', 'CellType')
  
  print("Writing output to file")
  
  if (nchar(dir_out) > 0) {
    dir.create(dir_out, showWarnings = FALSE)
    fout_scrna <- paste0(dir_out, '/', fout_prefix, 'scRNA_data.txt')
    fout_labels <- paste0(dir_out, '/', fout_prefix, 'cell_type_labels.txt')
  } else {
    fout_scrna <- paste0(fout_prefix, 'scRNA_data.txt')
    fout_labels <- paste0(fout_prefix, 'cell_type_labels.txt')
  }
  
  # Use write_delim instead of write.table for faster output
  print("Writing output to file")
  fwrite(scrna_count, fout_scrna, sep = '\t', quote = FALSE, row.names = FALSE)
  fwrite(cell_type_labels, fout_labels, sep = '\t', quote = FALSE, row.names = FALSE)
  
  
  print("Done")
}



#' Prepare and Launch CytoSPACE from SPATA2 Object
#'
#' This function extracts spatial transcriptomics data from a SPATA2 object and generates all necessary input files for running [CytoSPACE](https://github.com/ludvigla/CytoSPACE). It also generates a bash script (`cytoSpace.sh`) to execute the CytoSPACE pipeline in a specified Conda environment.
#'
#' @param object A SPATA2 object containing spatial transcriptomics data.
#' @param Reference_Annotation A string specifying the column in the feature data of the SPATA2 object that contains cell-type-specific abundance scores (e.g., from Cell2Location).
#' @param CS_folder A character string specifying the output folder where CytoSPACE input files and the run script will be saved.
#' @param Ref_mat Path to the reference single-cell RNA-seq expression matrix (in tab-delimited `.txt` format).
#' @param Ref_lables Path to the reference single-cell RNA-seq cell type labels (in tab-delimited `.txt` format).
#' @param scale Optional. A numeric value for spatial scaling (default: 30). Currently unused inside this function but reserved for future integration.
#'
#' @return This function writes the following files to `CS_folder`:
#' \describe{
#'   \item{`counts.txt`}{Gene expression matrix extracted from the SPATA2 object.}
#'   \item{`coords.txt`}{Spatial coordinates of each spot.}
#'   \item{`Cell_Fraction.txt`}{Estimated tissue fraction based on `Reference_Annotation`, aligned to cell types in `Ref_lables`.}
#'   \item{`cytoSpace.sh`}{A bash script that launches the CytoSPACE tool using the above files.}
#' }
#'
#' @details
#' The function:
#' \itemize{
#'   \item Extracts count matrix and spatial coordinates from the SPATA2 object.
#'   \item Calculates and normalizes the total tissue fraction per cell type using `Reference_Annotation`.
#'   \item Ensures alignment between SPATA2-derived fractions and reference cell types.
#'   \item Writes all required input files for CytoSPACE and a bash script to run it via `conda run -n cytospace ...`.
#' }
#'
#' @note This function assumes that the `cytospace` environment is already installed and accessible via `conda run -n cytospace`.
#'
#' @importFrom SPATA2 getCountMatrix getCoordsDf getFeatureDf
#' @importFrom dplyr select rename colSums
#' @importFrom data.table fwrite
#' @importFrom tibble rownames_to_column
#' @importFrom utils write.table
#'
#' @examples
#' \dontrun{
#' returnCytoSpace(
#'   object = spata_obj,
#'   Reference_Annotation = "cell2location_results",
#'   CS_folder = "./cytospace_input/",
#'   Ref_mat = "./reference/scRNA_matrix.txt",
#'   Ref_lables = "./reference/scRNA_labels.txt"
#' )
#' }
#'
#' @export
returnCytoSpace <- function(object, Reference_Annotation, CS_folder, Ref_mat, Ref_lables, scale=30){
  
  counts_st <- SPATA2::getCountMatrix(object) %>% as.data.frame() %>% rownames_to_column("V1")
  coords <- SPATA2::getCoordsDf(object) %>% dplyr::select(barcodes, row, col) %>% dplyr::rename("SpotID":=barcodes)
  
  #Tissue Fraction
  df <- SPATA2::getFeatureDf(object)
  
  if(any(c(Reference_Annotation %in% names(df))==F)) stop("Need to run cell2location with new data")
  
  tissue_fraction <- df %>% dplyr::select({{Reference_Annotation}}) %>% colSums() %>% as.data.frame()
  names(tissue_fraction) <- c("Fraction")
  tissue_fraction <- tissue_fraction %>% rownames_to_column("Index")
  
  total <- sum(tissue_fraction$Fraction)
  tissue_fraction$Fraction <- tissue_fraction$Fraction/total
  tissue_fraction <- t(tissue_fraction)
  
  ## Align Tissue fraction with ref
  ref_lab <- read.csv(Ref_lables, sep="\t")$CellType %>% unique()
  to_remove <- which(c(tissue_fraction[1,] %in% ref_lab)==F)
  
  if(length(to_remove)>2){tissue_fraction <- tissue_fraction[,-to_remove]}else{
    tissue_fraction=tissue_fraction
  }
  #Create a folder for the data
  #sample <- SPATA2::getSampleName(object)
  sample_dir <- CS_folder
  if(dir.exists(sample_dir)==F){dir.create(sample_dir)}
  
  #output files
  write.table(counts_st, file=paste0(sample_dir, "/counts.txt"), row.names=F, sep="\t", quote=F)
  write.table(coords, file=paste0(sample_dir, "/coords.txt"), row.names=F, sep="\t", quote=F)
  write.table(tissue_fraction, file=paste0(sample_dir, "/Cell_Fraction.txt"), row.names=T,col.names = F, sep="\t", quote=F)
  
  
  #create the conda run script
  
  
  run <- paste0("conda run -n cytospace_v1.1.0 cytospace --scRNA-path '" , Ref_mat, 
                "' --cell-type-path '", Ref_lables,
                "' --st-path '", paste0(sample_dir, "/counts.txt"), 
                "' --coordinates-path '", paste0(sample_dir, "/coords.txt"), 
                "' -o '", sample_dir,
                "' --cell-type-fraction-estimation-path '", paste0(sample_dir, "/Cell_Fraction.txt"), "' --solver-method 'lap_CSPR'" )
  
  
  string.start <- " #!/usr/bin/env bash -l \n \n \n "
  run.code <- paste0(run, " \n ")
  string.out <- base::paste0(string.start,"\n", paste0(run.code, collapse = "\n"), "\n \n print(\"Done\") \n \n")
  base::writeLines(string.out, paste0(CS_folder, "/cytoSpace.sh"))
}




importCytoSpace <- function(object, Reference_Annotation, CS_folder, Ref_mat, Ref_lables, scale=30){
  
  message(paste0(Sys.time(), "--- Create single cell SPATA2 object -----"))
  sc_loc <- read.csv(paste0(CS_folder,"/assigned_locations.csv")) %>% mutate(barcodes = SpotID) %>% select(-SpotID)
  
  ## Add x and y
  sc_loc <- left_join(sc_loc, SPATA2::getCoordsDf(object) %>% select(barcodes, x, y))
  
  
  # jitter position
  sc_loc <- sc_loc %>% left_join(., sc_loc %>% group_by(barcodes) %>% count())
  #scale=30
  sc_loc <- map_dfr(.x=sc_loc$barcodes %>% unique(), .f=function(.x){
    a <- sc_loc %>% filter(barcodes==.x)
    if(a$n[1]==1){a$x_sc=a$x; a$y_sc=a$y}else{
      a$x_sc <- runif(nrow(a), unique(a$x)-0.6*scale, unique(a$x)+0.6*scale) %>% sample()
      a$y_sc <- runif(nrow(a), unique(a$y)-0.6*scale, unique(a$y)+0.6*scale) %>% sample()
    }
    return(a)
  }, .progress=T)
  
  #object@data_add$cytospace = sc_loc
  
  
  return(sc_loc)
}


