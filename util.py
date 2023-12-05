import constants as const

def get_call_id(example):
    for k in const.CALL_MAP:
        if(example.call_type in const.CALL_MAP[k]):
            return const.CALL_IDS[k]