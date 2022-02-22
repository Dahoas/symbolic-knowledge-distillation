import gdown
import zipfile

def download(url: str, output: str) -> None:
	gdown.download(url, output, quiet=False)

def unzip(output: str) -> None:
	with zipfile.ZipFile(output, 'r') as zip_ref:
		zip_ref.extractall('.')

if __name__=='__main__':
	url = 'https://drive.google.com/file/d/1mUgv6fBp6uLU6PoSmfahF07hCjHdNYP_/view?usp=sharing'
	output = 'comet_distil.zip'
	download(url, output)
	unzip(output)