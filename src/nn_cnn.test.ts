import { zeros } from './factories';
import * as nn from './nn';
import { Tensor } from './tensor';
import { createFromSize } from './utils';


test("Conv2d Backward", () => {
    const input = new Tensor(createFromSize([1,3,20,30]),"float32");
    input.requiresGrad = true;
    const conv2d = new nn.Conv2d(3, 256, 3, 1, 0, "float32");
    const output = conv2d.forward(input);
    // output.backward();
});

test("Linear Backward", () => {
    const input = new Tensor(createFromSize([10]),"float32");
    input.requiresGrad = true;
    const linear = new nn.Linear(10, 256, "float32")
    const linear2 = new nn.Linear(256,1,"float32")
    let output = linear.forward(input);
    output = linear2.forward(output)
    output.backward()
});
