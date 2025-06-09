package fusion;

import org.nd4j.linalg.api.ndarray.INDArray;

import fusion.Layer.layerType;

public class Linear extends Module{
  public Layer layer;

  public Linear(String path){
    layer = new Layer(path, layerType.Linear);
  }

  public INDArray forward(INDArray x){
    return layer.Linear(x);
  }

  public INDArray forward(INDArray x, INDArray t){
    return layer.Linear(x);
  }
}
