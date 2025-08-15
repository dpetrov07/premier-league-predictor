import axios from 'axios';
import { FaSpinner, FaArrowLeft, FaArrowRight } from "react-icons/fa"
import React, { useEffect, useState } from "react";
import "./App.css";

function App() {
  const [upcomingMatches, setUpcomingMatches] = useState([]);
  const [selectedMatch, setSelectedMatch] = useState(null);
  const [prediction, setPrediction] = useState(null);

  const [loadingData, setLoadingData] = useState(false);

  const [currentPage, setCurrentPage] = useState(1);
  const matchesPerPage = 10;

  useEffect(() => {
    setLoadingData(true);
    // Gets future matches immediately on app render
    axios.get(`/api/future-matches?page=${currentPage}&page_size=${matchesPerPage}`)
      .then((response) => setUpcomingMatches(response.data))
      .catch((error) => console.error(error))
      .finally(() => setLoadingData(false));
  }, [currentPage]);

  const handleMatchClick = (match) => {
    // Gets prediction data for selected match from model
    if (loadingData) return;
    if (selectedMatch && selectedMatch.id === match.id) {
      setSelectedMatch(null);
      setPrediction(null);
      return;
    }

    // Sets prediction data and handles spam or bad requests
    setLoadingData(true);
    setSelectedMatch(match);
    axios.post("/api/predict-match", { matchId: match.id })
    .then((response) => {
      setPrediction(response.data);
    })
    .catch((error) => {
      console.error(error);
      setPrediction(null);
    })
    .finally(() => setLoadingData(false));
  }

  return (
    <div className="container">
      <h1 className="header">Premier League Predictor</h1>

      <section className="section">
        <h2 className="section-title">Upcoming Matches</h2>
        <table className="matches-table">
          <thead>
            <tr>
              <th>Date</th>
              <th>Time</th>
              <th>Team</th>
              <th>Opponent</th>
            </tr>
          </thead>
          <tbody>
            {upcomingMatches.map(({ id, date, time, team, opponent }, i) => (
              <React.Fragment key={i}>
                {/* Creates list of future matches with match data */}
                <tr
                  onClick={() => handleMatchClick({ id, team, opponent })}
                  style={{ cursor: "pointer" }}
                >
                  <td>{date}</td>
                  <td>{time}</td>
                  <td>{team}</td>
                  <td>{opponent}</td>
                </tr>

                {/* Creates dropdown box below selected match with prediction percentages */}
                {selectedMatch && selectedMatch.id === id && prediction && (
                  <tr className="prediction-row">
                    <td colSpan="4">
                      <div className="prediction-box">
                        {/* Displays loading sign before data is fetched */}
                        {loadingData ? (
                          <FaSpinner className="spinner" />
                        ) : (
                          <>
                            <span>
                              <strong>{team} Win:</strong>{" "}
                              {prediction.home_win_prob}%
                            </span>
                            <span>
                              <strong>Draw:</strong> {prediction.draw_prob}%
                            </span>
                            <span>
                              <strong>{opponent} Win:</strong>{" "}
                              {prediction.away_win_prob}%
                            </span>
                          </>
                        )}
                      </div>
                    </td>
                  </tr>
                )}
              </React.Fragment>
            ))}
          </tbody>
        </table>
        <div className="page-buttons">
          {/* Previous button and disables at first page */}
          <button
            onClick={() => setCurrentPage((prev) => Math.max(prev - 1, 1))}
            disabled={currentPage === 1}
          >
            <FaArrowLeft />
          </button>
          <span className="page-indicator">Page {currentPage}</span>
          {/* Next button and disables at last page */}
          <button
            onClick={() => setCurrentPage((prev) => prev + 1)}
            disabled={upcomingMatches.length < matchesPerPage}
          >
            <FaArrowRight />
          </button>
        </div>
      </section>
    </div>
  );
}

export default App;
