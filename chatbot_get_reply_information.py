from typing import ItemsView
import get_movie_info

# Get API Information from User Message
def get_title(title: str, nation, curPage, itmePerPage):
    try:
        title, start_year, end_year, check = divide_year(title)
        info_list = None

        if check:
            results = get_movie_info.get_info(title=title, start_year=start_year, end_year=end_year, 
                            nation_name=nation, curPage=curPage, itemPerPage=itmePerPage)
        else:
            results = get_movie_info.get_info(title, nation_name=nation, curPage=curPage, itemPerPage=itmePerPage)

        if len(results) != 0:
            info_list = summary_information(results)

        return info_list
    
    except Exception as e:
        print(e)
        return None


def get_actor(actor: str, curPage, itmePerPage):
    try:
        actor = " ".join(actor.split())
        results = get_movie_info.get_actor_info(actor=actor, curPage=curPage, itemPerPage=itmePerPage)

        if len(results) != 0:
            info_list = []
            
            for result in results:
                info = ""
                info += f"이름: {result['peopleNm']}\n"
                info += f"영어 이름: {result['peopleNmEn']}\n"
                info += f"역할: {result['repRoleNm']}\n"

                info += "Filmography: "
                filmo_list = result['filmoNames'].split("|")
                for i, f in enumerate(filmo_list):
                    info += f
                    if i != len(filmo_list) - 1:
                        info += ", "

                info_list.append(info)

            return info_list

        else:
            return None

    except Exception as e:
        print(e)
        return None


def get_director(director: str, nation, curPage, itmePerPage):
    try:
        director, start_year, end_year, check = divide_year(director)
        info_list = None

        if check:
            results = get_movie_info.get_info(director=director, start_year=start_year, end_year=end_year, 
                            nation_name=nation, curPage=curPage, itemPerPage=itmePerPage)
        else:
            results = get_movie_info.get_info(director=director, nation_name=nation, curPage=curPage, itemPerPage=itmePerPage)

        if len(results) != 0:
            info_list = summary_information(results)

        return info_list
    
    except Exception as e:
        print(e)
        return None


def get_rank(date: str, nation):
    try:
        date = " ".join(date.split())
        date = date.split(",")
        
        if len(date) > 1:
            if "한국" in date[1]:
                nation = "K"
            elif "외국" in date[1]:
                nation = "F"

        date = date[0]

        results = get_movie_info.get_ranking(date_str=date, nation_type=nation)
        info_list = None

        if len(results) != 0:
            info_list = []

            for result in results:
                info = ""
                info += f"순위: {result['rank']}\n"
                info += f"제목: {result['movieNm']}\n"
                info += f"개봉일: {result['openDt']}\n"
                info += f"신규여부: {'신규' if result['rankOldAndNew'] == 'NEW' else '기존'}\n"
                info += f"전일대비: {result['rankInten']}\n"
                info += f"누적 관객수: {result['audiAcc']}\n"

                info_list.append(info)

            return info_list

        else:
            return info_list

    except Exception as e:
        print(e)
        return None


def get_year(year: str, nation, curPage, itmePerPage):
    try:
        year = year.split("-")
        info_list = None

        if len(year) == 2:
            results = get_movie_info.get_info(start_year=year[0], end_year=year[1], 
                            nation_name=nation, curPage=curPage, itemPerPage=itmePerPage)
        else:
            results = get_movie_info.get_info(start_year=year[0], end_year=year[0], 
                            nation_name=nation, curPage=curPage, itemPerPage=itmePerPage)

        if len(results) != 0:
            info_list = summary_information(results)

        return info_list
    
    except Exception as e:
        print(e)
        return None


def get_country(country: str, curPage, itmePerPage):
    try:
        country, start_year, end_year, check = divide_year(country)
        info_list = None

        if check:
            results = get_movie_info.get_info(nation_name=country, start_year=start_year, end_year=end_year, 
                            curPage=curPage, itemPerPage=itmePerPage)
        else:
            results = get_movie_info.get_info(nation_name=country, curPage=curPage, itemPerPage=itmePerPage)

        if len(results) != 0:
            info_list = summary_information(results)

        return info_list
    
    except Exception as e:
        print(e)
        return None


def divide_year(text: str):
    text = " ".join(text.split())
    check = False

    if "," in text:
        text_list = text.split(",")
        if len(text_list) > 2:
            raise Exception
        
        text = text_list[0].strip()
        year = "".join(text_list[1].split())
        year = year.split("-")

        if len(year) > 2:
            raise Exception
        if len(year) == 1:
            start_year = year[0]
            end_year = year[0]
        else:
            start_year = year[0]
            end_year = year[1]

        check = True
        return text, start_year, end_year, check   

    else:
        return text, None, None, check


def summary_information(results):
    info_list = []

    for result in results:
        info = ""
        info += f"제목: {result['movieInfo']['movieNm']}\n"
        info += f"개봉일: {hyphen_date(result['movieInfo']['openDt'])}\n"
        info += f"상영시간: {result['movieInfo']['showTm']} minute\n"

        info += "국가: "
        for i, n in enumerate(result['movieInfo']['nations']):
            info += n['nationNm']
            if i != len(result['movieInfo']['nations']) - 1:
                info += ", "
            else:
                info += "\n"
        if len(result['movieInfo']['nations']) == 0:
            info += "\n"
        
        info += "장르: "
        for i, n in enumerate(result['movieInfo']['genres']):
            info += n['genreNm']
            if i != len(result['movieInfo']['genres']) - 1:
                info += ", "
            else:
                info += "\n"
        if len(result['movieInfo']['genres']) == 0:
            info += "\n"
        
        info += "감독: "
        for i, n in enumerate(result['movieInfo']['directors']):
            info += n['peopleNm']
            if i != len(result['movieInfo']['directors']) - 1:
                info += ", "
            else:
                info += "\n"
        if len(result['movieInfo']['directors']) == 0:
            info += "\n"
        
        info += "배우: "
        for i, n in enumerate(result['movieInfo']['actors']):
            info += n['peopleNm']
            if i != len(result['movieInfo']['actors']) - 1:
                info += ", "
        
        info_list.append(info)
    
    return info_list

# yyyymmdd -> yyyy-mm-dd
def hyphen_date(date: str):
    date_split = [date[:4], date[4:6], date[6:]]
    date = "-".join(date_split)

    return date