import datetime

SCRIPT_LIST = {
"unknown" : ("죄송합니다. 잘 모르겠어요.", "죄송합니다. 정확한 요구 사항이 필요해요!", "죄송합니다. 다시 입력해 주시겠어요?", 
"죄송해요. 사용법은 매뉴얼, 사용법 등으로 입력해 주세요."),

"hello" : ("Hello! 안녕하세요! 얻고 싶은 정보를 입력해 주세요. 사용법은 매뉴얼, 사용법 등으로 입력해 주세요.", 
"반갑습니다. 원하시는 정보를 입력해 주세요. 사용법은 매뉴얼, 사용법 등으로 입력해 보세요."),

"manual" : ("영화 제목, 배우, 감독, 순위, 연도, 국가를 이용해서 영화 정보를 얻을 수 있습니다.\nex) 배우 찾아줘, 영화 추천좀 해줘", 
"영화 제목으로 찾을래, 인기 있는 영화좀 등을 입력해 보세요.\n영화 제목, 배우, 감독, 순위, 연도, 국가를 이용해서 영화 정보를 얻을 수 있습니다."),

"title" : ("찾고자 하는 영화 제목을 입력해 주세요.\n검색어가 포함된 제목의 영화를 볼 수 있습니다.\nex) 컨저링\n연도와 함께 검색도 가능합니다.\nex) 컨저링,2013-2021 또는 컨저링,2013", 
"정보를 얻을 영화의 제목을 입력해 주세요. 영화 제목에 검색어가 포함된 영화를 볼 수 있습니다.\nex) 인질\n연도 입력도 가능합니다.\nex) 인질,2020-2021 또는 인질,2020"),

"actor" : ("찾고자 하는 영화 배우를 입력해 주세요. 배우가 출연한 영화 제목을 얻을 수 있습니다.\nex) 하정우", 
"배우가 출연한 영화 제목을 얻을 수 있어요! 영화 배우를 입력해 주세요.\nex) 하정우"),

"director" : ("찾고자 하는 영화 감독을 입력해 주세요. 영화 정보를 얻을 수 있습니다.\nex) 스티븐 스필버그\n연도 입력도 가능합니다.\nex) 스티븐 스필버그,2000-2005 또는 스티븐 스필버그,2020", 
"감독이 맡은 영화 정보를 얻을 수 있습니다. 찾고자 하는 영화 감독을 입력해주세요.\nex) 봉준호\n연도와 함께 검색도 가능합니다.\nex) 봉준호,2020-2021 또는 봉준호,2020"),

"rank" : (f"최근 영화 순위입니다.", "최근 인기있는 영화 순위입니다."),

"year" : ("연도별 개봉한 영화 정보를 얻을 수 있어요. 연도를 입력해 주세요.\nex) 2021 or 2020-2021", 
"특정 연도에 개봉한 영화 정보를 얻을 수 있어요. 연도를 입력해 주세요.\nex) 2021 or 2020-2021"),

"country" : ("국가별 영화 정보를 조회할 수 있습니다. 국가를 입력해 주세요.\nex) 한국\n연도 입력도 가능합니다.\nex) 한국,2004-2005 또는 한국,2020", 
"국가를 입력해 주세요. 해당 국가의 영화 정보를 얻을 수 있습니다.\nex) 미국,2010-2013 또는 미국,2018"),

"welcome" : ("어서 오세요. 영화 정보 챗봇입니다.\n사용법은 매뉴얼, 사용법 등으로 입력해 보세요!", 
"반갑습니다. 영화 정보 챗봇입니다.\n사용법은 매뉴얼, 사용법 등으로 입력해 보세요!", 
"안녕하세요! 영화 정보 챗봇입니다.\n다양한 영화 정보를 얻으세요. 사용법은 매뉴얼, 사용법 등으로 입력해 보세요!",
"어서 오세요! 영화 정보 챗봇입니다.\n여러 영화 정보를 얻을 수 있어요! 사용법은 매뉴얼, 사용법 등으로 입력해 보세요!"),

"other" : ("다른 정보가 더 필요한가요? 얻고 싶은 정보를 입력해 주세요.\n사용법은 매뉴얼, 사용법 등으로 입력해 주시면 됩니다.",
"더 필요하신 정보는 없나요? 원하시는 다른 정보가 있으시면 입력해 주세요.\n사용법은 매뉴얼, 사용법 등으로 입력해 주시면 됩니다."),

"none" : ("죄송합니다. 찾으시는 정보가 없어요.\n입력 형식이 잘못되었거나 데이터가 존재하지 않아요.",
"죄송해요. 정보를 찾을 수 없어요.\n데이터가 존재하지 않거나 입력 형식을 확인해 주세요.",
"정보가 존재하지 않아요. 입력 형식을 다시 한 번 확인해 주시겠어요?",
"찾으시는 정보가 없어요. 입력 형식 확인을 해주세요."),

"next" : ("다음 정보를 계속 보시겠어요? (ㅇ 또는 y)", 
"검색 된 다음 정보를 계속 보시겠어요? (ㅇ 또는 y)", 
"다음 정보가 존재하면 계속 확인하실 수 있어요. 계속 하시겠어요? (ㅇ 또는 y)"),

"rank_next" : ("특정 날짜의 순위도 보실 수 있어요!\nex)20201210\n한국, 외국 지정도 가능합니다.\nex)20170515, 한국\n(취소하시려면 ㄴ 또는 n)", 
"날짜를 지정해 순위를 볼 수 있어요!\nex)20170511\n한국, 외국 지정도 가능해요.\nex)20160813, 외국\n(취소하시려면 ㄴ 또는 n)"),

"end" : ("검색 결과가 존재하지 않아요.\n다른 정보가 더 필요하시면 얻고 싶은 정보를 입력해 주세요.", 
"더 이상 결과가 존재하지 않아요.\n필요하신 다른 정보를 입력해 주세요~"),

"cancel" : ("취소 되었어요. 얻고 싶은 정보를 입력해 주세요~", "취소 되었습니다. 원하시는 정보를 입력해 주세요.")
}