from PIL import Image

images = [
    # Image.open("images/1.jpg").convert("RGB"),
    # Image.open("images/2.jpg").convert("RGB"),
]

if __name__ == "__main__":
    import pandas as pd

    # Read dataset
    df = pd.read_csv('./datasets/Menu Items.csv').sample(n=10000)

    df = df[['Section', 'Item', 'Description', 'Price']] # Pick specific columns

    # Rename columns
    df = df.rename(columns={
        'Section' : 'category',
        'Item' : 'item',
        'Description' : 'description',
        'Price' : 'price',
    })

    df = df.dropna(subset=['item', 'description']) # drop rows which are null or empty in these columns
    df['category'] = df['category'].fillna('unknown') # fill unknown as value to null values
    df['price'] = df['price'].fillna('') # fill empty with null data

    # type caste all the values to string
    df['category'] = df['category'].astype(str)
    df['item'] = df['item'].astype(str)
    df['description'] = df['description'].astype(str)
    df['price'] = df['price'].astype(str)

    # clean spaces
    df['category'] = df['category'].str.strip()
    df['item'] = df['item'].str.strip()
    df['description'] = df['description'].str.strip()
    df['price'] = df['price'].str.strip()

    df = df[df["item"].str.len() > 2]
    df = df[df["description"].str.len() > 10]

    df['training_text'] = (
        '### Item: ' + df['item'] + '\n' +
        '### Details:\n' +
        'Category: ' + df['category'] + '\n' +
        'Description: ' + df['description'] + '\n' +
        'Price: ' + df['price'] + '\n' +
        '<|endoftext|>\n'          # explicit stop signal
    )


    text = '\n'.join(df['training_text'].tolist())

    with open("input.txt", 'w', encoding='utf-8') as f:
        f.write(text)