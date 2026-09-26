from rapidfuzz import fuzz
s1_a = "175 boulevard franklin roosevelt bordeaux nouvelle aquitaine"
c_a = "66 bis rue royale lille hauts de france"
print("addr_token_set:", fuzz.token_set_ratio(s1_a, c_a)/100.0)

s1_n = "alpha software solutions"
c_n = "zenith dental care"
print("name_token_set:", fuzz.token_set_ratio(s1_n, c_n)/100.0)
