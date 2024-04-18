path = "test_data/SERRF example dataset.xlsx"
sheet = 0

test = read_serff_format_data(path, sheet)

for i in "e", "f", "e_matrix", "p":
    temp_r = pd.read_table(f"test_data/{i}.tsv").fillna("na")
    temp_r = temp_r.reset_index(drop=True)
    temp_r.columns = [i.replace(".", "_") for i in temp_r.columns]
    tempdf = test[i].reset_index(drop=True).fillna("na")

    if not temp_r.equals(tempdf):
        print(f"{i} is different")
        break
