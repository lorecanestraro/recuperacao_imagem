CBIR with Spatial Weighting — Intel Image Classification

Este projeto implementa um sistema de Recuperação de Imagens Baseada em Conteúdo (CBIR) utilizando o dataset Intel Image Classification. 
O sistema combina descritores visuais clássicos (textura, forma e cor) com uma estratégia de janelas deslizantes (Sliding Window) e ponderação 
espacial baseada em IoU (Intersection over Union) para otimizar a precisão da busca por regiões de interesse.

🚀 Funcionalidades
Extração Multidescritor: 
-Combina histogramas HOG (forma/bordas), LBP (textura), histograma HSV e momentos de cor (média, desvio padrão e assimetria).
-Propostas de Região: Geração automática de caixas delimitadoras (Bboxes) multi-escala baseadas na estrutura da cena (Céu, Meio, Chão).
-Busca Híbrida (Visual + Espacial): Score final calculado via $Score = \alpha \cdot \text{CosSim} + \beta \cdot \text{IoU}$.
-Otimização de Hiperparâmetros: Sweep automático do parâmetro $\alpha$ para maximizar o mAP (Mean Average Precision).
-Relatórios Automáticos: Geração de um Dashboard visual em PNG e um relatório analítico detalhado em PDF com curvas Precision-Recall.
