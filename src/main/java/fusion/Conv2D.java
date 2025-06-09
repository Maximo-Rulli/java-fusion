package fusion;

import org.nd4j.linalg.api.ndarray.INDArray;

import fusion.Layer.layerType;

public class Conv2D extends Module{
  public Layer layer;
  public int s;
  public int p;

  public Conv2D(String path){
    layer = new Layer(path, layerType.Conv);
    s = 1;
    p = 0;
  }

  public Conv2D(String path, int stride, int padding){
    layer = new Layer(path, layerType.Conv);
    s = stride;
    p = padding;
  }

  public INDArray forward(INDArray x, INDArray t){
    return layer.Conv(x, s, p);
  }
  
  public INDArray forward(INDArray x){
    return layer.Conv(x, s, p);
  }
}
