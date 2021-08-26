import re

def problem1(searchstring):
    number =re.sub("\D","",searchstring)
    if (len(number) == 10) | (len(number) == 7):
        if len(searchstring) == 8:
            first_type = re.findall("\d{3}[-]\d{4}", searchstring)
            if len(first_type) > 0:
                return True
            else:
                return False
        elif len(searchstring) == 12:
            second_type = re.findall("\d{3}[-]\d{3}[-]\d{4}", searchstring)
            if len(second_type) > 0:
                return True
            else:
                return False
        elif len(searchstring) == 14:
            third_type = re.findall("\(\d{3}\)\s\d{3}[-]\d{4}", searchstring)
            if len(third_type) > 0:
                return True
            else:
                return False
        else:
            return False
    else:
        return False



        
def problem2(searchstring):

    addr = re.compile(r'\d+ (([A-Z][a-z]* )+)(Ave.|St.|Dr.|Rd.)')
    return addr.search(searchstring).group(1)[:-1]
    
def problem3(searchstring):

    addr = re.compile(r'(\d+) (([A-Z][a-z]* )+)(Ave.|St.|Dr.|Rd.)')
    return addr.sub(lambda x : x.group(1) + " " + (x.group(2)[:-1])[::-1] + " " + x.group(4),searchstring)



if __name__ == '__main__' :
    print(problem1('765-494-4600')) #True
    print(problem1(' 765-494-4600 ')) #False
    print(problem1('(765) 494 4600')) #False
    print(problem1('(765) 494-4600')) #True    
    print(problem1('494-4600')) #True
    
    print(problem2('The EE building is at 465 Northwestern Ave.')) #Northwestern
    print(problem2('Meet me at 201 South First St. at noon')) #South First
    
    print(problem3('The EE building is at 465 Northwestern Ave.'))
    print(problem3('Meet me at 201 South First St. at noon'))
