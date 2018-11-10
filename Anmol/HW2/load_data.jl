using CSV, Dates, DataFrames, Statistics, QuantEcon

function loaddata()
cd("C:\\Users\\jgsla\\Google Drive\\ECON_8185\\Anmol\\HW2\\Data")
#Loading Data
private = CSV.read("real_investment.csv")
gov = CSV.read("gov_investment.csv")
Hdata = CSV.read("labor.csv")
Ydata = CSV.read("GDP.csv")

#Renaming Columns

rename!(Hdata,  [Symbol("hours/population") => :Labor])
rename!(Ydata,  [:GDPC1 => :GDP])
rename!(private,  [:GPDIC1 => :investment])
#Same starting and ending year
private = private[Hdata[:DATE][1].<=private[:DATE].<=Hdata[:DATE][end],:]
Ydata = Ydata[Hdata[:DATE][1].<=Ydata[:DATE].<=Hdata[:DATE][end],:]
gov = gov[Hdata[:DATE][1].<=gov[:DATE].<=Hdata[:DATE][end],:]


#DATA will be a DataFrame containing all variables we buiild it from Ndata since it is monthly
DATA = copy(Hdata)
DATA[:GDP]=Ydata[:GDP]./DATA[:Population] .* 1000000 #In thousands of USD per capita
DATA[:Investment] = (gov[:Real_Investment]+private[:investment])./DATA[:Population] .* 1000000 #In thousands of USD
DATA[:GOV] = (gov[:Real_Consumption])./DATA[:Population] .* 1000000
#Some useful labor means
meanN = log(mean(DATA[:Labor]))

#We will use logs only
DATA[:Labor] = log.(DATA[:Labor])

DATA[:GDP] = log.(DATA[:GDP])
DATA[:GOV] = log.(DATA[:GOV])
DATA[:Investment] = log.(DATA[:Investment])



#Now, we have our dataset. We need to use the HP filter to remove trends
#in capital and GDP


DATA[:GDP_dev],DATA[:GDP_trend] = hp_filter(DATA[:GDP],1600)
DATA[:Investment_dev],DATA[:Investment_trend] = hp_filter(DATA[:Investment],1600)
DATA[:GOV_dev],DATA[:GOV_trend] = hp_filter(DATA[:GOV],1600)
#In our model, and in Data, labor has no trend so we assume its Steady State value is its mean.
DATA[:Labor_trend] = ones(282).* meanN
DATA[:Labor_dev] = DATA[:Labor] - DATA[:Labor_trend]
cd("C:\\Users\\jgsla\\Google Drive\\ECON_8185\\Anmol\\HW2")
return DATA
end

#If HP filter is desired, uncomment below:
#DATA[:Labor_dev],DATA[:Labor_trend] = hp_filter(DATA[:Labor],1600)

#=
plot(
    plot([DATA[:GDP],DATA[:GDP_trend]],legend = :bottomright, label = ["GDP","TREND"]),
    plot([DATA[:GDP_dev]], label = ["GDP Deviations"],legend = :bottomright),
    plot([DATA[:Investment],DATA[:Investment_trend]],legend = :bottomright, label = ["Investment","TREND"]),
    plot([DATA[:Investment_dev]],legend = :bottomright, label = ["Investment Deviations"])

    )

plot(      plot([DATA[:Labor],DATA[:Labor_trend]],legend = :bottomright, label = ["Labor","TREND"]),
        plot([DATA[:Labor_dev]],legend = :bottomright, label = ["Labor Deviations"]),
        plot([DATA[:GOV],DATA[:GOV_trend]],legend = :bottomright, label = ["Government Expenditure","TREND"]),
        plot([DATA[:GOV_dev]],legend = :bottomright, label = ["Government Deviations"])
    ) =#
