import configparser
import re

# import sys
# sys.path.insert(1, '/home/lyhourt.te/source/synthesis/text2num')


# from text.util import get_whole_word_regx
from text2num.util.file import get_dir_path

############# SETTING #############
config = configparser.ConfigParser()
config.read("%s/num2word.ini" % get_dir_path(__file__))

COMMON = config["COMMON"]
NUM = config["NUM"]
NUM_TY = config["NUM_TY"]
NUM_UNIT = config["NUM_UNIT"]
# MONTH           = config['MONTH']
NUM_OPERATOR = config["NUM_OPERATOR"]
DIGIT = config["DIGIT"]
TEL_PREFIX = config["TEL_PREFIX"]

############# PATTERN #############

NUM_EN_REGX = r"([0-9]+([.,][0-9]+)?)"
NUM_KM_REGX = r"([០-៩]+([.,][០-៩]+)?)"
NUM_KM_EXP = r"([-+]? ?[.,០-៩]+( [+-/x*] [.,០-៩]+)*)"

DATE_KM_REGX = r"([០-៩]+ ?[-\/] ?[០-៩]+ ?[-\/] ?[០-៩]+)"
TIME_KM_REGX = r"([០-៩]{1,2} ?: ?[០-៩]{1,2}( ?: ?[០-៩]{1,2})?)"


############# NUMBER #############

# 0: 6112: 48, ៩: 6121: 57


def en2km(num_en: str):
    if num_en is None:
        return None

    result = ""
    for c in num_en:
        if 48 > ord(c) > 57 and c != "." and c != "," and c != "-" and c != "+":
            raise Exception("Invalid en number: ", num_en)
        if c == ".":
            result += ","
        elif c == ",":
            continue
        elif c == "+" or c == "-":
            result += c
        else:
            try:
                result += chr(ord(c) + 6064)
            except Exception as e:
                print("Error convert num: ", num_en)
                raise e
    return result


# cover khmer number to english numbers
def km2en(num_km: str):
    # if num_km is None:
    #     return None

    result = ""
    for c in num_km:
        if 6112 > ord(c) > 6121 and c != "," and c != "." and c != "-" and c != "+":
            raise Exception("Invalid km number: ", num_km)
        if c == ",":
            result += "."
        elif c == ".":
            continue
        elif c == "+" or c == "-":
            result += c
        else:
            try:
                result += chr(ord(c) - 6064)
            except Exception as e:
                print("Error convert num: ", num_km)
                raise e
    return result


def is_num_km(num_km: str):
    return bool(re.match(r"[០-៩]+([,.][០-៩]+)*", num_km))


def is_num_en(num_en: str):
    return bool(re.match(r"[0-9]+([,.][0-9]+)*", num_en))


def is_float_km(num_km: str):
    return "," in num_km


def is_float_en(num_km: str):
    return "." in num_km


def digits2word(digits_km: str):
    try:
        result = []
        digits_en = km2en(digits_km)
        for num in digits_en:
            result.append(NUM[num])
        return " ".join(result)
    except Exception as e:
        print(f"Error convert digits to word: {digits_km}", e)
        raise e


def get_num_units(num_en: str, first_zero=False):
    if num_en is None or num_en == "":
        return None, None

    num_en = num_en.replace(",", "")

    # prepare
    point = num_en.find(".")
    if point != -1:
        num = int(num_en[:point])
        num_point = num_en[point + 1 :]  # str
    else:
        num = int(num_en)
        num_point = None

    # number part
    if num == 0:
        unit = [0]
    else:
        i = 0
        unit = []
        while num > 0:
            unit.append(num % 10)
            i += 1
            num = num // 10

        if first_zero is True:
            match = re.match("^[0]+", num_en)
            if match is not None:
                zeros = match.group(0)
                for z in zeros:
                    unit = unit + [0]

    # float part
    unit_point = None if num_point is None else []
    if num_point is not None:
        for num in num_point:
            unit_point.append(int(num))

    return unit[::-1], unit_point  # number part need to be reversed


# number to word
def int2word(num_km: str, first_zero=False):
    try:
        num_en = km2en(num_km)

        if num_en is None or num_en == "":
            return ""

        if is_float_en(num_en):
            raise Exception("num_km must not be float: ", num_km)

        result = []

        # sign number
        if num_en[0] == "+" or num_en[0] == "-":
            result.append(NUM_OPERATOR[num_en[0]])

            num_en = num_en[1:]
            num_km = num_km[1:]

        skip_unit = 0  # to skip number e.g. 1000000 -> 000000
        num_units, _ = get_num_units(num_en, first_zero)
        print("skip_unit before loop", skip_unit)
        for idx, value in enumerate(num_units):
            print("skip_unit in loop", skip_unit)
            if skip_unit > 0:
                skip_unit -= 1
                continue
            nums = num_units[idx:]
            # print('nums:',nums)

            # for number between 1 to 2 digit
            if len(nums) == 1:
                # basic digit
                result.append(NUM[str(value)])
            elif nums == [0] * len(nums):
                # if sub number is 00000 stop as the previous step already converted the number
                break
            elif len(nums) == 2:
                num_str = "".join([str(n) for n in nums])

                if num_str[0] == "0":
                    # first zero
                    num = NUM.get(str(value))
                    result.append(num)
                    continue
                else:
                    # get number from NUM and NUM_TY
                    num = NUM.get(num_str)
                    if num is None:
                        num = NUM_TY.get(num_str)
                    if num is not None:
                        result.append(num)
                        break
                    else:
                        # if not exists split the ty number: 25 -> 20, 5 will be converted next step
                        num_value = NUM_TY[str(value) + "0"]
                        # print('num_value',num_value)
                        result.append(num_value)

            # number bigger than 3 digit
            else:
                # get number from NUM_UNIT in form of 10^n
                num_unit = NUM_UNIT.get(str(pow(10, len(num_units) - idx - 1)))
                print("num-unit", num_unit)
                if num_unit is None:
                    sub_num = [str(value)]
                    print("nums:", nums)
                    for sub_idx, sub_value in enumerate(nums[1:]):
                        sub_num_unit = NUM_UNIT.get(
                            str(pow(10, len(nums) - sub_idx - 1))
                        )
                        print("sub_num_unit", sub_num_unit)
                        if sub_num_unit is not None:
                            num_value = num2word(num_km[: len(sub_num)])
                            result.append("%s %s" % (num_value, sub_num_unit))

                            if first_zero is False:
                                # define number of zero to be skipped. e.g. 100000000 -> 0000000
                                match = re.match("^[0]+", num_en[sub_idx:])
                                if match is not None:
                                    zeros = match.group(0)
                                    skip_unit = len(zeros) + len(sub_num) - 1
                                    print("skip_unit 1", skip_unit)
                            break
                        else:
                            sub_num.append(str(sub_value))
                            print("sub_num_unit e", sub_num_unit)
                else:
                    ##
                    num_value = NUM[str(value)]
                    result.append(
                        "%s %s" % (num_value, num_unit)
                    )  # remove compound word %s_%s

                    if first_zero is False:
                        # define number of zero to be skipped. e.g. 20012 -> 00
                        sub_value = num_en[idx + 1 :]
                        match = re.match("^[0]+", sub_value)
                        if match is not None:
                            zeros = match.group(0)
                            skip_unit = len(zeros)
                            print("skip_unit 2", skip_unit)

        return " ".join(result)
    except Exception as e:
        print(f"Error convert num to words: {num_km}", e)
        raise e


# number to word
def num2word(num_km):

    # didnt get call
    if type(num_km) != str:
        num_km = num_km.group(0)

    # didnt get call
    if num_km is None or num_km == "":
        return ""

    # remove white space
    num_km = re.sub(r"\s+|\.", "", num_km)

    # check if float base on ,
    if "," in num_km:
        # splite float base on ,
        parts = num_km.split(",")

        decimal = int2word(parts[0])
        point = int2word(parts[1], first_zero=True)

        return "%s %s %s" % (decimal, COMMON["FLOAT_POINT"], point)
    else:
        print("call int2word num_km ", num_km)
        word = int2word(num_km)
        return word


############# PHONE #############
def is_tel_num(num_km):
    num_en = km2en(num_km)
    if len(num_en) < 8 or "." in num_en or "," in num_en:
        return False

    prefix = num_en[:3]
    for company, tels in TEL_PREFIX.items():
        if prefix in tels:
            return True
    return False


############# DATE #############

# def is_date_km(date_km: str):
#     return bool(re.match(DATE_KM_REGX, date_km))


# def date2word(date_km: str, full_word=False):
#     """" dd/mm/yyyy, dd-mm-yyyy """

#     date_km = re.sub(r'\s+', '', date_km)
#     if date_km is None or date_km == '':
#         return ''

#     if not is_date_km(date_km):
#         raise Exception(f'Invalid km number: {date_km}')

#     parts = date_km.split('/')
#     if len(parts) == 1:
#         parts = date_km.split('-')

#     day = num2word(parts[0])
#     if full_word is True:
#         day = '%s %s' % (COMMON['DAY'], day)

#     month = MONTH.get(km2en(parts[1]))
#     if month is None:
#         raise Exception(f'Invalid km month: {month} - {date_km}')
#     if full_word is True:
#         month = '%s %s' % (COMMON['MONTH'], month)

#     year = num2word(parts[2])
#     if full_word is True:
#         year = '%s %s' % (COMMON['YEAR'], year)

#     return '%s %s %s' % (day, month, year)

# def date2text(text: str, full_word=True):
#     date_kms = re.findall(DATE_KM_REGX, text)
#     for date_km in date_kms:
#         word = date2word(date_km, full_word)
#         text = re.sub(get_whole_word_regx(date_km), '%s' % word, text, 1)

############# TIME #############

# def is_time_km(time_km: str):
#     return bool(re.match(TIME_KM_REGX, time_km))


# def time2word(time_km: str):
#     """" hh:mm:ss """

#     time_km = re.sub(r'\s+', '', time_km)
#     if time_km is None or time_km == '':
#         return ''

#     if not is_time_km(time_km):
#         raise Exception(f'Invalid km time: {time_km}')

#     parts = time_km.split(':')

#     # hour
#     result = num2word(parts[0])
#     result = COMMON['TIME_HH'] + ' ' + result

#     if len(parts) >= 2:
#         mm = num2word(parts[1])
#         result += ' %s %s' % (mm, COMMON['TIME_MN'])
#     if len(parts) >= 3:
#         ss = num2word(parts[2])
#         result += ' %s %s' % (ss, COMMON['TIME_SS'])

#     return result

# def time2text(text: str):
#     time_kms = re.findall(TIME_KM_REGX, text)
#     for time_km, _ in time_kms:
#         if time_km.startswith(COMMON['TIME_HH']):
#             time = time_km.replace(COMMON['TIME_HH'], '').strip()
#             time = time2word(time)
#         else:
#             time = time2word(time_km)

#         text = re.sub(r'\b%s' % time_km, '%s' % time, text, 1)

#######


def num_en2km(text):
    if type(text) != str:
        text = text.group(0)
    num_ens = re.findall(NUM_EN_REGX, text)
    for num_en, num_point in num_ens:
        num = en2km(num_en)
        text = text.replace(num_en, "%s" % num, 1)
    return text


def num2text(text: str):
    num_km_exps = re.findall(NUM_KM_EXP, text)
    print("num_km_exps : ", num_km_exps)
    for num_km_exp, _ in num_km_exps:
        exps = num_km_exp.split()

        new_exp = ""
        for exp in exps:
            if bool(re.match(r"[+-/x*]", exp)):
                new_exp += " " + NUM_OPERATOR[exp]
            else:
                if is_tel_num(exp):
                    # TODO refactor when pos-tagging exist
                    new_exp += " " + digits2word(exp)
                else:
                    new_exp += " " + num2word(exp)

        text = text.replace(num_km_exp, " %s " % new_exp, 1)
        print("text: ", text)

