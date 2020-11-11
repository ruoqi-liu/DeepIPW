1. Cohort.csv contains all the patients with at least 1 year insurance enrollment history after 1st time CAD encounter.
   - ENROLID: 511,274 unique patients’ id
   - ADMDATE: 1st CAD admission date for the inpatient records
   - SVCDATE: 1st CAD service date for the outpatient records
   - Index_date: The date of first CAD encounter – i.e., min (ADMDATE, SVCDATE)
   - DTSTART: Date of insurance enrollment start
   - DTEND: Date of insurance enrollment end

2. demo.csv contains the demographics information of all the patients.
   - ENROLID: 511,274 unique patients’ id
   - DOBYR: birth year
   - SEX: gender: 1- male; 2- female
   - MSA: Metropolitan Statistical Area – city mappings
   - REGION: Geographic Region of employee residence – region mappings
 
3. There are 6 drug tables from year 2012 to year 2017
   - ENROLID: Patient id
   - NDCNUM: Drug id
   - RXINGID: Omop_Rx_ing_concept_ID 
   - SVCDATE: Date to take the prescription
   - DAYSUPP: Days supply: The number of days of drug therapy covered by this prescription
 
4. There are 6 inpatient tables from year 2012 to year 2017
   [SPECIAL Note: Schema changed on and after year 2015. There is a new field DXVER! Before 2015, all codes are ICD9. After 2015, codes are mixture of ICD-9 and ICD-10]
   [SPECIAL NOTE: there are 39 codes that are duplicated between ICD-10-CM and ICD-9-CM: https://www.miramedgs.com/web/17-the-code-newsletter/mmgs-the-code-may-2015-issue/225-duplicate-icd-10-cm-and-icd-9-cm-codes]
   - ENROLID: Patient id
   - DX1-DX15: diagnosis codes
   - DXVER: “9” = ICD-9-CM and “0” = ICD-10-CM
   - PROC1-PROC15: Procedure codes
   - ADMDATE: Admission date for this inpatient visit
   - Days: The number of days stay in the inpatient hospital

5. There are 6 outpatient tables from year 2012 to year 2017
   [SPECIAL Note: Schema changed on and after year 2015. There is a new field DXVER! Before 2015, all codes are ICD9. After 2015, codes are mixture of ICD-9 and ICD-10]
   - ENROLID: Patient id
   - DX1-DX4: diagnosis codes
   - DXVER: “9” = ICD-9-CM and “0” = ICD-10-CM
   - PROC1: a procedure code
   - PROCTYP: *: ICD-9-CM; 0: ICD-10-CM; 1: CPT; 3: UB92 Revenue Code 6: NABSP; 7: HCPC; 8: CDT (ADA)
   - SVCDATE: Service date for this outpatient visit
