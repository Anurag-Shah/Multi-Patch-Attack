import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets
import torchvision.transforms
from pretrained_models_pytorch import pretrainedmodels

# Config
IMAGE_SIZE = 300     # h, w in pixels. 
PATCH_PROPORTION = 0.017    # proportion of patch to original image. 0.05 is ideal for single patch
ENABLE_CUDA = True
MAX_PATCHES = 3    # Maximum number of patches applied. Above 3 causes issues
EPOCHS = 15
TARGET_CLASS = 890 # 859 seems to be the toaster class
EPOCH_SIZE = 2000 # Train/Test size in epoch. Edit this to reduce train size per epoch
LOGGING_INTERVAL = 10 # Sample logging interval
LOGGING_DIR = "logging_directory"
IMDIR = 'imagenetdata/val'

# Model. can be changed freely to anything trained on imagenet. Might require resizing.
resnet = pretrainedmodels.__dict__['resnet50'](num_classes=1000, pretrained='imagenet')

# Globals to initialize later
train_loader = None
test_loader = None
max_out = None
min_out = None

def main():
    global train_loader, test_loader, max_out, min_out, IMAGE_SIZE
    try:
        os.makedirs(LOGGING_DIR)
    except OSError:
        pass
    torch.backends.cudnn.benchmark = True
    
    if ENABLE_CUDA:
        resnet.cuda()

    train_loader, test_loader = generate_dataloaders()

    min_in, max_in = resnet.input_range[0], resnet.input_range[1]
    min_in, max_in = np.array([min_in, min_in, min_in]), np.array([max_in, max_in, max_in])
    mean, std = np.array(resnet.mean), np.array(resnet.std) 
    min_out, max_out = np.min((min_in - mean) / std), np.max((max_in - mean) / std)
    temp = IMAGE_SIZE ** 2
    temp = temp * PATCH_PROPORTION
    temp = int(temp ** (0.5))
    patch = np.random.rand(1, 3, temp, temp)
    patch_shape = patch.shape
    
    for epoch in range(1, EPOCHS):
        patch = train(epoch, patch, patch_shape)
        test(epoch, patch, patch_shape)

def generate_dataloaders():
    train_transforms = torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop(224), torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    test_transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(size=(224, 224)), .transforms.RandomHorizontalFlip(), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    index = np.arange(50000)
    np.random.shuffle(index)
    train_index = index[:EPOCH_SIZE]
    test_index = index[EPOCH_SIZE: (2 * EPOCH_SIZE)]
    train_dataset = torchvision.datasets.ImageFolder(root=IMDIR, transform=train_transforms)
    test_dataset = torchvision.datasets.ImageFolder(root=IMDIR, transform=test_transforms)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, sampler=torch.utils.data.sampler.SubsetRandomSampler(train_index), num_workers=2, pin_memory=True, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, sampler=torch.utils.data.sampler.SubsetRandomSampler(test_index), num_workers=2, pin_memory=True, shuffle=False)
    return train_loader, test_loader

def train(epoch, patch, patch_shape):
    resnet.eval()
    successful_attacks = 0
    total_attacks = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        if batch_idx % LOGGING_INTERVAL == 0:
            print(f"Epoch {epoch} Sample {batch_idx}")

        if ENABLE_CUDA:
            data = data.cuda()
            labels = labels.cuda()
        data, labels = torch.autograd.Variable(data), torch.autograd.Variable(labels)
        prediction = resnet(data)
        
        # transform path
        data_shape = data.data.cpu().numpy().shape
        patch, mask, opt_mask = square_transform(patch, data_shape, patch_shape)
        patch = torch.FloatTensor(patch)
        mask = torch.FloatTensor(mask)
        opt_mask = torch.FloatTensor(opt_mask)
        if ENABLE_CUDA:
            patch = patch.cuda()
            mask = mask.cuda()
            opt_mask = opt_mask.cuda()
        patch = torch.autograd.Variable(patch)
        mask = torch.autograd.Variable(mask)
        opt_mask = torch.autograd.Variable(opt_mask)
 
        adv_x, patch = attack(data, patch, mask)
        
        attacked_label = resnet(adv_x).data.max(1)[1][0]
        old_label = labels.data[0]
        
        total_attacks += 1
        if attacked_label == TARGET_CLASS:
            successful_attacks += 1
            if batch_idx % LOGGING_INTERVAL == 0:
                torchvision.utils.save_image(adv_x.data, f"./{LOGGING_DIR}/{epoch}_{batch_idx}.png", normalize=True)
 
        masked_patch = torch.mul(opt_mask, patch)
        patch = masked_patch.data.cpu().numpy()
        new_patch = np.zeros(patch_shape)
        for i in range(new_patch.shape[0]):
            for j in range(new_patch.shape[1]):
                smx, smy = np.nonzero(patch[i][j])
                new_patch[i][j] = patch[i][j][smx.min():smx.max() + 1, smy.min():smy.max() + 1]
 
        patch = new_patch
    print(f"Epoch {epoch} successful proportion: {successful_attacks / (total_attacks + 0.0)}")
    return patch

def test(epoch, patch, patch_shape):
    resnet.eval()
    successful_attacks = 0
    total_attacks = 0
    for batch_idx, (data, labels) in enumerate(test_loader):
        if ENABLE_CUDA:
            data = data.cuda()
            labels = labels.cuda()
        data, labels = torch.autograd.Variable(data), torch.autograd.Variable(labels)
        prediction = resnet(data)
        
        data_shape = data.data.cpu().numpy().shape
        patch, mask, opt_mask = square_transform(patch, data_shape, patch_shape)
        patch = torch.FloatTensor(patch)
        mask = torch.FloatTensor(mask)
        opt_mask = torch.FloatTensor(opt_mask)
        if ENABLE_CUDA:
            patch = patch.cuda()
            mask = mask.cuda()
            opt_mask = opt_mask.cuda()
        patch = torch.autograd.Variable(patch)
        mask = torch.autograd.Variable(mask)
        opt_mask = torch.autograd.Variable(opt_mask)
 
        adv_x = torch.mul((1 - mask),data) + torch.mul(mask, patch)
        adv_x = torch.clamp(adv_x, min_out, max_out)
        
        attacked_label = resnet(adv_x).data.max(1)[1][0]

        total_attacks += 1
        if attacked_label == TARGET_CLASS:
            successful_attacks += 1
       
        masked_patch = torch.mul(opt_mask, patch)
        patch = masked_patch.data.cpu().numpy()
        new_patch = np.zeros(patch_shape)
        for i in range(new_patch.shape[0]):
            for j in range(new_patch.shape[1]):
                smx, smy = np.nonzero(patch[i][j])
                new_patch[i][j] = patch[i][j][smx.min():smx.max() + 1, smy.min():smy.max() + 1]
        patch = new_patch
    print(f"Test successful proportion: {successful_attacks / (total_attacks + 0.0)}")

def attack(x, patch, mask):
    resnet.eval()

    x_out = nn.functional.softmax(resnet(x))
    target_prob = x_out.data[0][TARGET_CLASS]

    adv_x = torch.mul((1 - mask), x) + torch.mul(mask, patch)
    
    count = 0 
    while target_prob < 0.9:
        count += 1
        adv_x = torch.autograd.Variable(adv_x.data, requires_grad=True)
        adv_out = nn.functional.log_softmax(resnet(adv_x))
        adv_out_probs, adv_out_labels = adv_out.max(1)
        Loss = -adv_out[0][TARGET_CLASS]
        Loss.backward()
        adv_grad = adv_x.grad.clone()
        adv_x.grad.data.zero_()
        patch -= adv_grad 
        
        adv_x = torch.mul((1 - mask), x) + torch.mul(mask, patch)
        adv_x = torch.clamp(adv_x, min_out, max_out)
 
        out = nn.functional.softmax(resnet(adv_x))
        target_prob = out.data[0][TARGET_CLASS]

        if count >= 1000:
            break
    return adv_x, patch

def square_transform(patch, data_shape, patch_shape):
    num_patches = random.randint(1, MAX_PATCHES)
    patch_0 = np.copy(patch)
    x = np.zeros(data_shape)
    coords = generate_coords(x.shape, patch_shape, num_patches)
    for i in range(x.shape[0]):
        # rotate
        rot = np.random.choice(4)
        for j in range(patch_0[i].shape[0]):
            patch_0[i][j] = np.rot90(patch_0[i][j], rot)
        # extract x/y location
        random_x, random_y = coords[0]
       
        x[i][0][random_x:random_x + patch_shape[-1], random_y:random_y + patch_shape[-1]] = patch_0[i][0]
        x[i][1][random_x:random_x + patch_shape[-1], random_y:random_y + patch_shape[-1]] = patch_0[i][1]
        x[i][2][random_x:random_x + patch_shape[-1], random_y:random_y + patch_shape[-1]] = patch_0[i][2]

    # Generate optimized patch mask. This is used to actually propagate the mask onwards
    opt_mask = np.copy(x)
    opt_mask[opt_mask != 0] = 1.0

    # Apply further patches.
    for k in range(1, num_patches):
        patch_i = np.copy(patch)
        for i in range(x.shape[0]):
            # rotate
            rot = np.random.choice(4)
            for j in range(patch_i[i].shape[0]):
                patch_i[i][j] = np.rot90(patch_i[i][j], rot)
            # extract x/y location
            random_x, random_y = coords[k]
           
            x[i][0][random_x:random_x + patch_shape[-1], random_y:random_y + patch_shape[-1]] = patch_i[i][0]
            x[i][1][random_x:random_x + patch_shape[-1], random_y:random_y + patch_shape[-1]] = patch_i[i][1]
            x[i][2][random_x:random_x + patch_shape[-1], random_y:random_y + patch_shape[-1]] = patch_i[i][2]
    
    mask = np.copy(x)
    mask[mask != 0] = 1.0
    
    # The "patch" (var x) is now the entire initial input image taken from resnet_val with the patches overlaid
    # The mask contains the patch masks. This is important because the entire base patch image is optimized, 
    #   so that all patches can be optimized at the same time. So the masks let us fetch the patches afterwards
    return x, mask, opt_mask

def generate_coords(x_shape, patch_shape, it):
    coords = []
    for i in range(it):
        random_x = np.random.choice(IMAGE_SIZE)
        random_y = np.random.choice(IMAGE_SIZE)
        while invalid_coords(random_x, random_y, coords, x_shape, patch_shape):
            random_x = np.random.choice(IMAGE_SIZE)
            random_y = np.random.choice(IMAGE_SIZE)
        coords.append((random_x, random_y))
    return coords

def invalid_coords(rx, ry, coords, x_shape, patch_shape):
    if rx + patch_shape[-1] > x_shape[-1] or ry + patch_shape[-1] > x_shape[-1]:
        return True
    for c in coords:
        #c[0] is x, c[1] is y
        x_covered = (rx >= c[0] - 1 and rx <= c[0] + patch_shape[-1] + 1) or (c[0] >= rx - 1 and c[0] <= rx + patch_shape[-1] + 1)
        y_covered = (ry >= c[1] - 1 and ry <= c[1] + patch_shape[-1] + 1) or (c[1] >= ry - 1 and c[1] <= ry + patch_shape[-1] + 1)
        if x_covered and y_covered:
            return True
    return False

if __name__ == "__main__":
    main()