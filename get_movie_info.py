import requests
import json

kofic_key = "40c4fa95d44dd845656317ab34c4eaaf"
kofic_title_url = "http://www.kobis.or.kr/kobisopenapi/webservice/rest/movie/searchMovieList.json"
kofic_actor_url = "http://www.kobis.or.kr/kobisopenapi/webservice/rest/people/searchPeopleList.json"
kofic_week_ranking_url = "http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchWeeklyBoxOfficeList.json"
kofic_common_url = "http://www.kobis.or.kr/kobisopenapi/webservice/rest/code/searchCodeList.json"

nation_code = {
    '한국': '22041011',
    '대만': '22041001',
    '말레이시아': '22041002',
    '북한': '22041003',
    '싱가포르': '22041004',
    '아프카니스탄': '22041005',
    '이란': '22041006',
    '인도': '22041007',
    '중국': '22041009',
    '태국': '22041010',
    '이스라엘': '22041013',
    '필리핀': '22041014',
    '아랍에미리트연합국정부': '22041015',
    '몽고': '22041016',
    '티베트': '22041017',
    '카자흐스탄': '22041018',
    '캄보디아': '22041019',
    '이라크': '22041020',
    '우즈베키스탄': '22041021',
    '베트남': '22041022',
    '인도네시아': '22041023',
    '카타르': '22041024',
    '기타_1': '22041099',
    '미국': '22042002',
    '멕시코': '22042001',
    '캐나다': '22042003',
    '자메이카': '22042004',
    '엘살바도르': '22042005',
    '트리니다드토바고': '22042006',
    '케이맨제도': '22042007',
    '기타_2': '22042099',
    '일본': '22041008',
    '베네수엘라': '22043001',
    '브라질': '22043002',
    '아르헨티나': '22043003',
    '콜롬비아': '22043004',
    '칠레': '22043005',
    '페루': '22043006',
    '우루과이': '22043007',
    '쿠바': '22043008',
    '기타_3': '22043099',
    '홍콩': '22041012',
    '그리스': '22044001',
    '네덜란드': '22044002',
    '덴마크': '22044003',
    '독일': '22044004',
    '러시아': '22044005',
    '벨기에': '22044006',
    '스웨덴': '22044007',
    '스위스': '22044008',
    '스페인': '22044009',
    '영국': '22044010',
    '오스트리아': '22044011',
    '이탈리아': '22044012',
    '체코': '22044013',
    '터키': '22044014',
    '포르투갈': '22044015',
    '폴란드': '22044016',
    '프랑스': '22044017',
    '핀란드': '22044018',
    '헝가리': '22044019',
    '불가리아': '22044020',
    '보스니아': '22044021',
    '크로아티아': '22044022',
    '노르웨이': '22044023',
    '에스토니아': '22044024',
    '아일랜드': '22044025',
    '잉글랜드': '22044026',
    '아이슬란드': '22044027',
    '루마니아': '22044028',
    '팔레스타인': '22044029',
    '세르비아': '22044030',
    '룩셈부르크': '22044031',
    '마케도니아': '22044032',
    '서독': '22044033',
    '소련': '22044034',
    '알바니아': '22044035',
    '유고슬라비아': '22044036',
    '몰타': '22044037',
    '우크라이나': '22044038',
    '슬로바키아': '22044039',
    '총괄(연감)': '22044099',
    '호주': '22045001',
    '뉴질랜드': '22045002',
    '피지': '22045003',
    '기타_5': '22045099',
    '남아프리카공화국': '22046001',
    '부탄': '22046002',
    '이집트': '22046003',
    '나이지리아': '22046004',
    '보츠와나': '22046005',
    '리비아': '22046006',
    '모로코': '22046007',
    '케냐': '22046008',
    '기타_6': '22046099',
    '기타_9': '22049999',
}


def from_title(title):
    title_params = {
        "key": kofic_key,
        "movieNm": title,
    }
    title_response = requests.get(url=kofic_title_url, params=title_params)
    if title_response.status_code == 200:
        data = title_response.json()
        return data['movieListResult']['movieList']
    else: return None


def from_actor(actor_name):
    actor_params = {
        "key": kofic_key,
        "peopleNm": actor_name
    }
    actor_response = requests.get(url=kofic_actor_url, params=actor_params)
    if actor_response.status_code == 200:
        data = actor_response.json()
        return data['peopleListResult']['peopleList']
    else: return None


def from_director(director_name):
    director_params = {
        "key": kofic_key,
        "directorNm": director_name,
    }
    director_response = requests.get(url=kofic_title_url, params=director_params)
    if director_response.status_code == 200:
        data = director_response.json()
        return data['movieListResult']['movieList']
    else: return None


def get_ranking(date_str):
    week_ranking_params = {
        "key": kofic_key,
        "targetDt": date_str,  # format: yyyymmdd (ex. 20210906)
        "weekGb": "0",
    }
    week_response = requests.get(url=kofic_week_ranking_url, params=week_ranking_params)
    if week_response.status_code == 200:
        data = week_response.json()
        return data['boxOfficeResult']['weeklyBoxOfficeList']
    else: return None


def from_opendate(start_year, end_year):  # format: yyyy (ex. 2021)
    opendate_params = {
        "key": kofic_key,
        "openStartDt": start_year,
        "openEndDt": end_year,
    }
    open_response = requests.get(url=kofic_title_url, params=opendate_params)
    if open_response.status_code == 200:
        data = open_response.json()
        return data['movieListResult']['movieList']

def from_nation(nation_name):
    if nation_code[nation_name] is not None:
        nation_params = {
            "key": kofic_key,
            "repNationCd": nation_code[nation_name],
        }
    else: return None

    nation_response = requests.get(url=kofic_title_url, params=nation_params)
    if nation_response.status_code == 200:
        data = nation_response.json()
        return data['movieListResult']['movieList']
    else: return None


def get_common_code():
    code_params = {
        "key": kofic_key,
        "comCode": "2204" # 국가코드
    }
    code_response = requests.get(url=kofic_common_url, params=code_params)
    if code_response.status_code == 200:
        data = code_response.json()
        print(data['codes'])

