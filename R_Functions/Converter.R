
#' Convert AnnData Object to SPATA2 Object
#'
#' This function takes an `AnnData` object (typically from Python via `reticulate`) and converts it into a `SPATA2` object, including optional integration of scVI expression data and spatial image/shape information.
#'
#' @param adata An AnnData object containing single-cell spatial transcriptomics data, typically imported from Python using `reticulate`.
#' @param sample_name A character string specifying the sample name to be assigned to the resulting SPATA2 object.
#' @param scVI Logical, whether to use the `X_exp` layer (typically used for scVI-normalized data) as an alternative expression matrix. Default is `FALSE`.
#' @param image Optional. Path to a spatial image file (e.g., histology image) associated with the sample.
#' @param shapes Optional. Path to a Parquet file containing shape data for spatial annotation (e.g., segmentation polygons).
#'
#' @return A SPATA2 object with added expression matrix, metadata, and optionally scVI matrix and shape/image data.
#'
#' @details 
#' - The function converts `adata$X` to a transposed matrix for use as the expression count matrix.
#' - Metadata from `adata$obs` is added as cell features.
#' - If `scVI = TRUE`, the expression matrix from `adata$layers[["X_exp"]]` is added and set as the active matrix.
#' - If `shapes` is provided, shape data is read using the `arrow` package.
#' - If `image` is provided, a message is shown, but no action is currently performed (you can extend this part).
#'
#' @importFrom SPATA2 initiateSpataObject_CountMtr addFeatures addExpressionMatrix setActiveExpressionMatrix
#' @importFrom dplyr select
#' @importFrom tibble rownames_to_column
#' @importFrom arrow read_parquet
#' 
#' @examples
#' \dontrun{
#' adata <- reticulate::import("anndata")$read_h5ad("my_file.h5ad")
#' spata_object <- ADATA2SPATA(adata, sample_name = "Sample1", scVI = TRUE)
#' }
#'
#' @export
ADATA2SPATA <- function(adata, sample_name, scVI = F, image=NULL, shapes=NULL){
  
  meta <- adata$obs %>% rownames_to_column("barcodes")
  coords <- meta %>% dplyr::select(barcodes,x,y)
  cellxgene <- adata$X %>% t()
  
  object <- 
    SPATA2::initiateSpataObject_CountMtr(coords_df = coords,
                                         count_mtr = cellxgene,
                                         sample_name = sample_name,
                                         ScaleData=T, RunPCA=F, FindNeighbors=F, FindClusters=F, RunTSNE=F, RunUMAP=F)
  
  object <- 
    SPATA2::addFeatures(object, meta)
  
  if(scVI==T){
    scVI = adata$layers[["X_exp"]] %>% t()
    object <- SPATA2::addExpressionMatrix(object, expr_mtr = scVI,  mtr_name= "scVI")
    object <- SPATA2::setActiveExpressionMatrix(object , "scVI")
  }
  
  
  #plotSurface(object, color_by="CD3D", display_image=F, use_scattermore=T)
  
  
  
  ## If shapes data are avaiable the slot shapes is the path to the shapes data
  if(!is.null(shapes)){
    message("Add shape data")
    shape_input = arrow::read_parquet(shapes)
  }
  
  ## If image data are avaiable the slot image is the path to the image data
  if(!is.null(image) ){
    message("Add Image data")
  }
  
  return(object)
  
}

#' Convert SPATA2 Object to AnnData Object
#'
#' This function converts a `SPATA2` object into a Python-compatible `AnnData` object using the `reticulate` interface. It utilizes Seurat and SingleCellExperiment as intermediaries and retains spatial coordinates.
#'
#' @param object A `SPATA2` object containing spatial transcriptomics data.
#'
#' @return An `AnnData` object (Python object) with gene expression matrix, metadata, and spatial coordinates stored in `obsm[["spatial"]]`.
#'
#' @details 
#' - Extracts count matrix from the SPATA2 object and converts it to a `SingleCellExperiment` via `Seurat`.
#' - Uses `reticulate` to call Python libraries (`scanpy`, `numpy`, and `pandas`) and construct the `AnnData` object.
#' - Adds spatial coordinates from the SPATA2 object to the `AnnData`'s `obsm[["spatial"]]` slot.
#'
#' @note Requires Python packages: `scanpy`, `numpy`, and `pandas` to be available in the active `reticulate` environment.
#'
#' @importFrom SPATA2 getBarcodes getCountMatrix getCoordsDf
#' @importFrom Seurat CreateSeuratObject as.SingleCellExperiment
#' @importFrom SingleCellExperiment colData rowData assay
#' @importFrom reticulate import
#' @importFrom dplyr %>%
#' 
#' @examples
#' \dontrun{
#' # Convert SPATA2 object to AnnData for use in Python
#' adata <- SPATA2ANDATA(spata_object)
#' }
#'
#' @export
SPATA2ANDATA <- function(object){
  
  library(SingleCellExperiment)
  library(anndata)
  sc <- reticulate::import("scanpy")
  np <- reticulate::import("numpy")
  pd <- reticulate::import("pandas")
  bc <- getBarcodes(object)
  sce <- 
    Seurat::CreateSeuratObject(counts=SPATA2::getCountMatrix(object)) %>% 
    Seurat::as.SingleCellExperiment()
  exprs <- assay(sce, "counts")
  col_data <- as.data.frame(colData(sce))
  row_data <- as.data.frame(rowData(sce))
  ## Create AnnData
  adata <- AnnData(X = t(exprs), obs = col_data, var = row_data)
  adata$obsm[["spatial"]] <- getCoordsDf(object)[,c("x", "y")] %>% as.matrix()
  return(adata)
}



#' Subset a SPATA2 Object by Barcodes
#'
#' This function subsets a `SPATA2` object by a given set of barcodes. It filters the expression data, coordinates, features, and trajectory projections to only include specified barcodes, and updates relevant metadata in the object.
#'
#' @param object A `SPATA2` object to be subset.
#' @param barcodes A character vector of barcodes (spot identifiers) to retain in the subset.
#' @param verbose Logical. If `TRUE`, print informative messages during processing. Default is `NULL` (no feedback unless explicitly set).
#'
#' @return A modified `SPATA2` object containing only the specified barcodes.
#'
#' @details 
#' - Filters the feature data frame (`feature_df`), spatial coordinates (`coords_df`), and expression matrices for each modality in `object@data`.
#' - Also subsets trajectory projections (`object@trajectories`) and updates barcode-related metadata.
#' - Tracks the number of subsetting operations and provides user feedback using `confuns::give_feedback`.
#'
#' @note Assumes that `hlpr_assign_arguments()` is available in your environment and used to capture function arguments internally.
#'
#' @importFrom dplyr filter mutate across %>%
#' @importFrom purrr map
#' @importFrom glue glue
#' @importFrom confuns give_feedback
#'
#' @examples
#' \dontrun{
#' subset_barcodes <- c("AAACCTGAGAGCTACC", "AAACCTGAGGCTCCTC")
#' new_spata_obj <- splitSPATA(spata_object, barcodes = subset_barcodes)
#' }
#'
#' @export
splitSPATA <- function (object, barcodes, verbose = NULL) {
  hlpr_assign_arguments(object)
  bcs_keep <- barcodes
  object <- getFeatureDf(object) %>% dplyr::filter(barcodes %in% 
                                                     {
                                                       {
                                                         bcs_keep
                                                       }
                                                     }) %>% dplyr::mutate(dplyr::across(.cols = where(base::is.factor), 
                                                                                        .fns = base::droplevels)) %>% setFeatureDf(object = object, 
                                                                                                                                   feature_df = .)
  object <- getCoordsDf(object) %>% dplyr::filter(barcodes %in% 
                                                    {
                                                      {
                                                        bcs_keep
                                                      }
                                                    }) %>% setCoordsDf(object, coords_df = .)
  object@data[[1]] <- purrr::map(.x = object@data[[1]], .f = ~.x[, 
                                                                 bcs_keep])
  object@trajectories[[1]] <- purrr::map(.x = object@trajectories[[1]], 
                                         .f = function(traj) {
                                           traj@projection <- dplyr::filter(traj@projection, 
                                                                            barcodes %in% {
                                                                              {
                                                                                bcs_keep
                                                                              }
                                                                            })
                                           return(traj)
                                         })
  object@information$barcodes <- object@information$barcodes[object@information$barcodes %in% 
                                                               bcs_keep]
  object@information[["subset"]][["barcodes"]] <- c(barcodes, 
                                                    object@information[["subset"]][["barcodes"]])
  if (base::is.numeric(object@information[["subsetted"]])) {
    object@information[["subsetted"]] <- object@information[["subsetted"]] + 
      1
  }
  else {
    object@information[["subsetted"]] <- 1
  }
  #object <- setTissueOutline(object, verbose = verbose)
  n_bcsp <- nBarcodes(object)
  confuns::give_feedback(msg = glue::glue("{n_bcsp} barcode spots remaining."), 
                         verbose = verbose)
  return(object)
}








