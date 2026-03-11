import numpy as np
from parameters import p
cimport constant_parameters as c
cimport draw_husband
cimport draw_wife
cimport calculate_wage
cimport meeting_partner
cimport libc.math as cmath
cdef extern from "randn.cc":
    double uniform()
    double maxvalue_filter(double arr[], int indexes[], int ilen)
from calculate_utility_single_women cimport calculate_utility_single_women
from calculate_utility_married cimport calculate_utility_married
from calculate_utility_single_men cimport calculate_utility_single_men


cdef int single_men(int t, double[:, :, :, :, :, :, :, :, :, :, :, :, :, :, :, :] w_emax,
    double[:, :, :, :, :, :, :, :, :, :, :, :, :, :, :, :] h_emax,
    double[:,:,:,:,:,:,:,:,:] w_s_emax,
    double[:,:,:,:,:,:,:,:,:] h_s_emax, verbose) except -1:
    cdef double[:] mother
    cdef int iter_count = 0
    cdef double sum_emax = 0
    cdef int married_index = -99
    cdef int choose_partner = 0
    cdef int school
    cdef int exp
    cdef int kids
    cdef int home_time
    cdef int ability
    cdef int mother_educ
    cdef int mother_marital
    cdef int draw
    cdef double wage_w_full
    cdef double wage_w_part
    cdef double wage_h_full
    cdef double wage_h_part
    cdef double single_women_value
    cdef double single_men_value
    cdef double[18] u_wife
    cdef double[18] u_husband
    cdef double[18] u_wife_full
    cdef double[18] u_husband_full
    cdef double[13] u_w_single_full
    cdef double[7] u_h_single_full

    if verbose:
        print("====================== single men:  ======================")
    cdef draw_husband.Husband husband = draw_husband.Husband()
    cdef draw_wife.Wife wife = draw_wife.Wife()
    husband.age = 17 + t
    # c.max_period, c.school_size, c.exp_size, c.kids_size, c.health_size, c.home_time_size, c.ability_size, c.mother_size, c.mother_size])
    for school in range(0, c.school_size):   # loop over school
        husband.schooling = school
        draw_husband.update_school(husband)
        for exp in range(0, c.exp_size):           # loop over experience
            husband.exp = c.exp_vector[exp]
            for kids in range(0, 2):                # for each number of kids: 0, 1,   - open loop of kids
                husband.kids = kids
                for home_time in range(0, c.home_time_size):       # home time loop - three options
                    husband.home_time_ar = c.home_time_vector     #c.home_time_vector[home_time]
                    for ability in range(0, 1): #c.ability_size):     # for each ability level: low, medium, high - open loop of ability
                        #husband.ability_i = ability
                        #husband.ability_value = c.ability_vector[ability] * p.sigma_ability_h  # husband.ability - low, medium, high
                        for mother_educ in range(0,c.mother_size):
                            husband.mother_educ = mother_educ
                            for mother_marital in range(0, c.mother_marital_size):
                                husband.mother_marital = mother_marital
                                draw_husband.update_ability_back(husband)
                                sum_emax = 0
                                iter_count = iter_count + 1
                                if verbose:
                                    print(husband)
                                for draw in range(0, c.DRAW_B):
                                    married_index = -99
                                    choose_partner = 0
                                    _, _, prob_full_h, prob_part_h, tmp_full_h = calculate_wage.calculate_wage_h(
                                        husband)
                                    single_men_value, single_men_index, _ = calculate_utility_single_men(
                                        h_s_emax, 0, 0, tmp_full_h, husband, t,
                                        u_h_single_full, 1)
                                    if husband.age < 20:
                                        prob_meet_potential_partner = cmath.exp(p.omega1) / (1.0 + cmath.exp(p.omega1))
                                    elif single_men_index == 6 and husband.schooling < 4:
                                        prob_meet_potential_partner = cmath.exp(p.omega2) / (1.0 + cmath.exp(p.omega2))
                                    else:
                                        prob_meet_potential_partner = meeting_partner.prob(husband.age)
                                    wife = draw_wife.draw_wife_back(husband, c.mother[0], c.mother[1], c.mother[2])

                                    _, _, prob_full_w, prob_part_w, tmp_full_w = calculate_wage.calculate_wage_w(
                                        wife)
                                    home_time_h, home_time_w, home_time_h_preg, home_time_w_preg =                                         calculate_utility_married(w_emax, h_emax, 0, 0, 0, 0,
                                                                  tmp_full_h, tmp_full_w, wife,
                                                                  husband, t, u_wife, u_husband,
                                                                  u_wife_full, u_husband_full,
                                                                  1)
                                    single_women_value, single_women_index, single_women_ar = calculate_utility_single_women(
                                        w_s_emax, 0, 0,
                                        tmp_full_w, wife, t, u_w_single_full, 1)

                                    weighted_utility = float('-inf')
                                    married_index = -99
                                    for i in range(0, 18):
                                        if u_wife[i] > single_women_value and u_husband[i] > single_men_value:
                                            if c.bp * u_wife[i] + (1 - c.bp) * u_husband[i] > weighted_utility:
                                                weighted_utility = c.bp * u_wife[i] + (1 - c.bp) * u_husband[i]
                                                married_index = i
                                    single_outside_option = prob_full_h * maxvalue_filter(u_h_single_full, [0, 2, 6], 3) +                                                            prob_part_h * maxvalue_filter(u_h_single_full, [0, 4, 6], 3) +                                                            (1 - prob_full_h - prob_part_h) * maxvalue_filter(u_h_single_full, [0, 6], 2)
                                    if married_index > -99:
                                        temp = prob_meet_potential_partner * u_husband[married_index] + (1 - prob_meet_potential_partner) * single_outside_option
                                    else:
                                        temp = single_outside_option
                                    sum_emax += temp

                                # end draw backward loop
                                h_s_emax[t][school][exp][kids][husband.health][home_time][ability][mother_educ][mother_marital] = sum_emax / c.DRAW_B
                                if verbose:
                                    print("emax(", t, ", ", school, ", ", exp,", ", kids, ",", ability, ")=", sum_emax / c.DRAW_B)
                                    print("======================================================")

    return iter_count
