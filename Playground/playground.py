# #16. You are a loyal customer and the store to offering discounts to their loyal customers:

#     between 0 & 50 CAD, you get 10 % discounts,
#     between 51 & 100 CAD, you get 20 % discounts,
#     more than 100 CAD, you get 40 % discounts,
#     Its Monday, and you are spending more than 50 CAD, you get a cherry on top! 10 CAD OFF

# write a function to calculate how much the customer will pay for any spending.
# Tricky?
# You need to develop the logic!
# Take a pen and a notebook and do the calculations if needed!
# test the function for [51,50,100,101] using for loop with you function


def effectivepay(totalspend, monday):

    if monday == False:
        if totalspend <= 50:
            return totalspend * 0.90
        elif totalspend > 50 and totalspend <= 100:
            return totalspend * 0.80
        elif totalspend > 100:
            return totalspend * 0.60
    else:
        if totalspend <= 50:
            return totalspend * 0.90
        elif totalspend > 50 and totalspend <= 100:
            return (totalspend-10) * 0.80
        elif totalspend > 100:
            return (totalspend-10) * 0.60

