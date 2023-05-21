import { zeros } from './factories';
import * as nn from './nn';
import { Tensor } from './tensor';
import { createFromSize } from './utils';


test("Conv2d Backward", () => {
    const input = new Tensor(createFromSize([1, 3, 20, 30]), "float32");
    input.requiresGrad = true;
    const conv2d = new nn.Conv2d(3, 256, 3, 1, 0, "float32");
    const output = conv2d.forward(input);
    // output.backward();
});

test("Linear training", () => {
    const input = new Tensor([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], "float32");
    const target = new Tensor([[123]], "float32")
    const linear = new nn.Linear(10, 20, "float32")
    const linear2 = new nn.Linear(20, 1, "float32")
    input.requiresGrad = true;
    for (let i = 0; i < 100; i++) {
        let t1 = linear.forward(input)
        let output = linear2.forward(t1);
        // console.log(`t1: ${t1.toArray()},output: ${output.toArray()}`)
        let loss = (output.sub(target)).square();
        console.log("loss: ",loss.toArray())

        // start update
        loss.backward()
        linear.BP()
        linear2.BP()

    }
});


test("Linear training", () => {
    const input = new Tensor([1], "float32");
    const target = new Tensor([[123]], "float32")
    const linear = new nn.Linear(1, 1, "float32")
    input.requiresGrad = true;
    // for (let i = 0; i < 200; i++) {
    //     let t1 = linear.forward(input)
    //     let loss = (t1.sub(target)).square();
    //     console.log("loss: ",loss.toArray())

    //     // start update
    //     loss.backward()
    //     linear.BP()

    // }
});

