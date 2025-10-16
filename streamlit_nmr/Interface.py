# app_rmn.py
import streamlit as st
import nmrglue as ng
import matplotlib.pyplot as plt
import os

st.title("Visualizador de Espectro ¹H-RMN")

# --- Upload do diretório de Bruker ---
st.write("Selecione a pasta `pdata` do seu espectro Bruker:")
pasta = st.text_input("Caminho da pasta (ex: C:/Users/vinic/Documents/.../pdata/1)")

if pasta:
    if os.path.exists(pasta):
        try:
            # --- Leitura do espectro Bruker ---
            dic, data = ng.bruker.read_pdata(pasta)
            udic = ng.bruker.guess_udic(dic, data, strip_fake=True)
            uc = ng.fileiobase.uc_from_udic(udic, dim=0)
            ppm = uc.ppm_scale()

            # --- Plot do espectro ---
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(ppm, data, color='black', lw=1)
            ax.invert_xaxis()  # eixo ppm decrescente
            ax.set_xlabel("Deslocamento químico (ppm)")
            ax.set_ylabel("Intensidade (u.a.)")
            ax.set_title("Espectro ¹H-RMN")

            st.pyplot(fig)

        except Exception as e:
            st.error(f"Erro ao ler os dados: {e}")
    else:
        st.warning("Caminho inválido. Verifique se a pasta existe.")
