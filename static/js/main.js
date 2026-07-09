document.addEventListener('DOMContentLoaded', () => {
    const predictionForm = document.getElementById('predictionForm');
    
    if (predictionForm) {
        predictionForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const submitBtn = document.getElementById('submitBtn');
            const originalText = submitBtn.innerText;
            submitBtn.innerText = 'Processing...';
            submitBtn.disabled = true;
            
            const formData = {
                age: document.getElementById('age').value,
                gender: document.getElementById('gender').value,
                income: document.getElementById('income').value,
                income_type: document.getElementById('income_type').value,
                years_employed: document.getElementById('years_employed').value,
                occupation: document.getElementById('occupation').value,
                car: document.getElementById('car').value,
                realty: document.getElementById('realty').value,
                children: document.getElementById('children').value,
                housing_type: document.getElementById('housing_type').value,
                family_status: document.getElementById('family_status').value,
                education: document.getElementById('education').value,
                work_phone: document.getElementById('work_phone').checked ? 1 : 0,
                phone: document.getElementById('phone').checked ? 1 : 0,
                email: document.getElementById('email').checked ? 1 : 0
            };
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(formData)
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    document.getElementById('resultSection').classList.remove('d-none');
                    
                    const alertBox = document.getElementById('alertBox');
                    
                    if (data.prediction === 'Approved') {
                        alertBox.className = 'alert alert-success fs-5 fw-bold';
                        alertBox.innerHTML = '<i class="bi bi-check-circle-fill me-2"></i> Application Approved';
                    } else {
                        alertBox.className = 'alert alert-danger fs-5 fw-bold';
                        alertBox.innerHTML = '<i class="bi bi-x-circle-fill me-2"></i> Application Declined';
                    }
                    
                    document.getElementById('resProb').innerText = data.approval_probability + '%';
                    
                    const riskEl = document.getElementById('resRisk');
                    riskEl.innerText = data.risk_level;
                    riskEl.className = data.risk_level === 'High' ? 'fw-bold text-danger m-0' : 
                                       data.risk_level === 'Medium' ? 'fw-bold text-warning m-0' : 'fw-bold text-success m-0';
                    
                    // Scroll to result
                    document.getElementById('resultSection').scrollIntoView({ behavior: 'smooth' });
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                alert('Request failed: ' + error.message);
            } finally {
                submitBtn.innerText = originalText;
                submitBtn.disabled = false;
            }
        });
    }
});
