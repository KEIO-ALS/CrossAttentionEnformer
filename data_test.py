from data.dataset_utils import load_basenji2
from tqdm import tqdm


trainloader, testloader, validloader = load_basenji2()
# pbar = tqdm(total=len(trainloader))
# for data in trainloader:
#     pbar.update()
# pbar.close()
# print(data[0].shape, data[1].shape, data[2])

pbar = tqdm(total=len(testloader))
for data in testloader:
    print(data[0].shape, data[1].shape, data[2])
    pbar.update()
pbar.close()


# pbar = tqdm(total=len(validloader))
# for data in validloader:
#     pbar.update()
# pbar.close()
# print(data[0].shape, data[1].shape, data[2].shape)