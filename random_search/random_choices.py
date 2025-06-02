def get_random_search_choices(user_value, default_list, around=1, scale=2):
    """
    user_value: 사용자가 직접 입력한 값 또는 None
    default_list: default로 쓸 리스트
    around: 수치형이면, 앞뒤 몇 개 값 생성
    scale: 배수(learning rate 등에서 유용)
    """
    if user_value is None:
        return default_list
    # 정수/실수면, user_value 기준으로 값 생성
    if isinstance(user_value, (int, float)):
        if isinstance(user_value, int):
            return list(sorted(set([user_value - around, user_value, user_value + around] + default_list)))
        else:  # float
            return list(sorted(set([user_value / scale, user_value, user_value * scale] + default_list)))
    
    # bool이나 카테고리면 그냥 [user_value] + default
    if isinstance(user_value, bool):
        return list(sorted(set([user_value] + default_list)))
    return default_list
