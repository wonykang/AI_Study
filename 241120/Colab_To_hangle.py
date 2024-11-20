# # @title
# # 나눔 폰트 설치
# # !apt -qq -y install fonts-nanum

# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm

# fontpath = 'C:\Users\UserC\Desktop\AILLM\fonts/NanumBareunGothicL.ttf'
# font = fm.FontProperties(fname=fontpath, size=10)
# # Add the font cache (use the public function)
# fm.fontManager.addfont(fontpath)

# # 그래프에 retina display 적용
# # 고사양이 필요한 것이라 여기서 안됨
# # %config InlineBackend.figure_format = 'retina'

# # Colab 의 한글 폰트 설정
# plt.rc('font', family='NanumBarunGothic')

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 폰트 경로를 절대 경로로 수정
fontpath = './fonts/NanumBarunGothic.ttf'  # 이 경로를 본인의 폰트 파일 위치로 수정
font = fm.FontProperties(fname=fontpath, size=10)

# 폰트 캐시 추가
fm.fontManager.addfont(fontpath)

# 그래프에 사용할 한글 폰트 설정
plt.rc('font', family='NanumBarunGothic')

# 예시 그래프 출력
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.title('나눔 고딕 폰트 테스트')  # 한글 제목
plt.show()
