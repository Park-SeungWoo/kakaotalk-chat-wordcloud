import pandas as pd

pd.set_option('display.max_rows', None)
from matplotlib import pyplot as plt
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
import numpy as np

from konlpy.tag import Okt
from stopwordsiso import stopwords
from collections import Counter

import emoji
import warnings

warnings.filterwarnings('ignore')


def remove_emoji(text):
    return emoji.replace_emoji(text, '')


def none_func(x):
    return 0


data_df = pd.read_csv('./datas/chats.csv')

data_df = data_df[['User', 'Message']]
data_df['User'] = data_df['User'].apply(remove_emoji)  # remove emojis in username
data_df['Message'] = data_df['Message'].apply(remove_emoji)  # remove emojis in messages
names = list(set(data_df['User']))  # get names

fir_chats = data_df[data_df['User'] == names[0]]

sec_chats = data_df[data_df['User'] == names[1]]

fir_texts = ' '.join(fir_chats['Message'])
sec_texts = ' '.join(sec_chats['Message'])

# preprocess text datas
tokenizer = Okt()
fir_tok = tokenizer.pos(fir_texts, norm=True)
sec_tok = tokenizer.pos(sec_texts, norm=True)

stopword_list = list(stopwords('ko'))
stopword_list.extend(['하다', '있다', '되다', '이다', '돼다', '않다', '그렇다', '아니다', '이렇다', '그렇다', '어떻다'])

## remove stopwords
fir_res = []
sec_res = []
for word in fir_tok:
    if word[1] not in ['Josa', 'Eomi', 'Punctuation', 'Foreign', 'KoreanParticle', 'Alpha']:  # filter by POS
        if (len(word[0]) != 1) & (
                word[0] not in stopword_list):
            fir_res.append(word[0])

for word in sec_tok:
    if word[1] not in ['Josa', 'Eomi', 'Punctuation', 'Foreign', 'KoreanParticle', 'Alpha']:  # filter by POS
        if (len(word[0]) != 1) & (
                word[0] not in stopword_list):
            sec_res.append(word[0])

# count freqs
fir_dict = Counter(fir_res)
sec_dict = Counter(sec_res)

fir = dict(fir_dict.most_common(200))
sec = dict(sec_dict.most_common(200))

# visualization
## graph
plt.rc('font', family='Nanum GaRamYeonGgoc')
# plt.figure(figsize=(15, 7))
# plt.tight_layout()
# plt.subplot(2, 1, 1)
# plt.bar(fir.keys(), fir.values())
# plt.title(names[0])
# plt.xticks(rotation=90)
#
# plt.subplot(2, 1, 2)
# plt.bar(sec.keys(), sec.values())
# plt.xticks(rotation=90)
# plt.title(names[1])
# plt.show()

## word cloud
mask = np.array(Image.open('./assets/Kakao_logo.jpg'))
coloring = ImageColorGenerator(mask)

word_cloud_fir = WordCloud(font_path='/Users/seungwoosmac/Library/Fonts/BMJUA_ttf.ttf',
                       mask=mask,
                       width=2000,
                       height=1000,
                       background_color='white',
                       ).generate_from_frequencies(fir)

word_cloud_sec = WordCloud(font_path='/Users/seungwoosmac/Library/Fonts/BMJUA_ttf.ttf',
                       mask=mask,
                       width=2000,
                       height=1000,
                       background_color='white',
                       ).generate_from_frequencies(sec)

plt.figure(figsize=(10, 10))
# plt.imshow(word_cloud_fir.recolor(color_func=coloring), interpolation='bilinear')  # use image's own color
plt.imshow(word_cloud_fir.recolor(colormap='spring'), interpolation='bilinear')  # use matplotlib colormap
plt.axis('off')
plt.tight_layout(pad=0)
plt.savefig(f'./assets/{names[0]}_chat_datas.png')
# plt.imshow(word_cloud_sec.recolor(color_func=coloring), interpolation='bilinear')  # use image's own color
plt.imshow(word_cloud_sec.recolor(colormap='spring'), interpolation='bilinear')  # use matplotlib colormap
plt.axis('off')
plt.tight_layout(pad=0)
plt.savefig(f'./assets/{names[1]}_chat_datas.png')