library('getopt')
command=matrix(c(
    'help', 'h', 0,'loical', '显示此帮助信息',
    'input', 'i', 1, 'character', '输入文件',
    byrow=T, ncol=5
))
library(TxDb.Hsapiens.UCSC.hg38.knownGene)
txdb <- TxDb.Hsapiens.UCSC.hg38.knownGene
exon_txdb=exons(txdb)
genes_txdb=genes(txdb)
o <- findOverlaps(exon_txdb, genes_txdb)    

t1 <- exon_txdb[queryHits(o)]          
t2 <- genes_txdb[subjectHits(o)]       
t1 <- as.data.frame(t1)
t1$geneid <- mcols(t2)[ ,1]       

g_l <- lapply(split(t1, t1$geneid), function(x){
  tmp <- apply(x, 1, function(y){
    y[2]:y[3]                    
  })
  length(unique(unlist(tmp)))   
})
head(g_l)                     
g_l <- data.frame(gene_id=names(g_l), length=as.numeric(g_l))  # 
options(connectionObserver = NULL)
library(org.Hs.eg.db)
s2g <- toTable(org.Hs.egSYMBOL)
s2e <- toTable(org.Hs.egENSEMBL)
head(s2e)
g_l_temp <- merge(g_l, s2g, by='gene_id')   
g_l_temp <- g_l_temp[!duplicated(g_l_temp$symbol_id),]
head(g_l_temp)
rownames(g_l_temp) <- g_l_temp [,3]
a <- read.table(command$input)
ng <- intersect(rownames(a), g_l_temp$symbol_id)   
exprSet <- a[ng, ]         
expr1 = exprSet/g_l_temp[ng, ]$length
tpm = t(t(expr1/colSums(expr1))) * 10^6

write.table(tpm, file = './logtpm_file.csv', sep=",")