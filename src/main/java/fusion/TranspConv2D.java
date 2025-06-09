package fusion;

import org.nd4j.linalg.api.ndarray.INDArray;

import fusion.Layer.layerType;

public class TranspConv2D extends Module{
  public Layer layer;
  public int s;
  public int p;

  public TranspConv2D(String path){
    layer = new Layer(path, layerType.Conv);
    s = 1;
    p = 0;
  }

  public TranspConv2D(String path, int stride, int padding){
    layer = new Layer(path, layerType.Conv);
    s = stride;
    p = padding;
  }

  public INDArray forward(INDArray x){
    return layer.TranspConv(x, s, p);
  }

  public INDArray forward(INDArray x, INDArray t){
    return layer.TranspConv(x, s, p);
  }
}