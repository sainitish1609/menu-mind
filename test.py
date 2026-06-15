from PIL import Image

images = [
    Image.open("images/1.jpg").convert("RGB"),
    Image.open("images/2.jpg").convert("RGB"),
]

with open('./datasets/en.openfoodfacts.org.products.tsv', 'r') as f:
    buffer = f.read()

print(buffer[100:])