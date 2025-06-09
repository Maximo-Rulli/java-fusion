package fusion;

import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class Module {
  public abstract INDArray forward(INDArray x, INDArray t);
}
