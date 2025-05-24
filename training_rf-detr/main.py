from rfdetr import RFDETRBase

model = RFDETRBase()

model.train(dataset_dir="dataset", epochs=50, batch_size=4, grad_accum_steps=4, lr=1e-4)