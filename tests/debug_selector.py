from dr_case.candidate_selector import CandidateSelector

selector = CandidateSelector.from_model_file(
    'models/som_optimized.pkl', 
    'data/unified_disease_symptom_data_full.json'
)

result = selector.select(['fever', 'headache', 'cough', 'fatigue'])

print(f'BMU: {result.bmu}')
print(f'BMU distance: {result.bmu_distance}')
print(f'Active units: {result.active_units}')
print(f'Memberships count: {len(result.memberships)}')
print(f'Cumulative mass: {result.cumulative_mass}')
print(f'Candidates: {result.candidate_count}')

# Перевіримо конфіг
print(f'\nConfig:')
print(f'  alpha: {selector.config.alpha}')
print(f'  k: {selector.config.k}')
print(f'  tau: {selector.config.tau}')
print(f'  lambda: {selector.config.membership_lambda}')